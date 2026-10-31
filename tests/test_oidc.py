"""core/oidc.py against a fake sign-in provider: its own signing keys, discovery, key set and token endpoint, all served
through httpx's mock transport, so no network is used."""
import base64
import json
import time

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec, rsa

from core import oidc

ISSUER = "https://idp.example.test"
CLIENT = "ragleap-client"
REDIRECT = "https://ragleap.example.test/auth/callback"
NONCE = "nonce-value-abcdefghijklmnop"


def _jwk(public_key, kid, alg):
    data = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(public_key) if alg.startswith(("RS", "PS"))
                      else jwt.algorithms.ECAlgorithm.to_jwk(public_key))
    data.update(kid=kid, use="sig", alg=alg)
    return data


class Provider:
    """A tiny fake provider. `keys` maps kid -> (private key, alg)."""

    def __init__(self):
        self.keys = {"k1": (rsa.generate_private_key(public_exponent=65537, key_size=2048), "RS256")}
        self.calls = []
        self.issuer_in_discovery = ISSUER
        self.token_status = 200
        self.token_body = None
        self.redirect_on_discovery = False

    def jwks(self):
        return {"keys": [_jwk(k.public_key(), kid, alg) for kid, (k, alg) in self.keys.items()]}

    def handler(self, request):
        self.calls.append((request.method, str(request.url)))
        url = str(request.url)
        if url.endswith("/.well-known/openid-configuration"):
            if self.redirect_on_discovery:
                return httpx.Response(302, headers={"location": "https://evil.example.test/"})
            return httpx.Response(200, json={"issuer": self.issuer_in_discovery, "authorization_endpoint": ISSUER + "/authorize",
                                             "token_endpoint": ISSUER + "/token", "jwks_uri": ISSUER + "/keys"})
        if url == ISSUER + "/keys":
            return httpx.Response(200, json=self.jwks())
        if url == ISSUER + "/token":
            self.last_form = dict(x.split("=", 1) for x in request.content.decode().split("&"))
            body = self.token_body if self.token_body is not None else {"id_token": "x.y.z", "access_token": "unused"}
            return httpx.Response(self.token_status, json=body)
        return httpx.Response(404)

    def token(self, kid="k1", key=None, alg=None, **claims):
        priv, key_alg = self.keys[kid] if key is None else (key, alg or "RS256")
        now = int(time.time())
        body = {"iss": ISSUER, "aud": CLIENT, "sub": "user-123", "iat": now, "exp": now + 300, "nonce": NONCE,
                "email": "Person@Example.Test", "email_verified": True, "name": "A Person", "amr": ["pwd", "otp"]}
        body.update(claims)
        body = {k: v for k, v in body.items() if v is not None}
        return jwt.encode(body, priv, algorithm=alg or key_alg, headers={"kid": kid})


@pytest.fixture
def idp(monkeypatch):
    oidc.reset_cache()
    provider = Provider()
    monkeypatch.setattr(oidc, "_make_client", lambda: httpx.Client(transport=httpx.MockTransport(provider.handler), follow_redirects=False))
    yield provider
    oidc.reset_cache()


def make_cfg(monkeypatch, **extra):
    env = {"OIDC_ISSUER": ISSUER, "OIDC_CLIENT_ID": CLIENT, "OIDC_CLIENT_SECRET": "x" * 12, "OIDC_REDIRECT_URL": REDIRECT}
    env.update(extra)
    return oidc.load_config(env)


def check(cfg, idp, raw, nonce=NONCE):
    return oidc.validate_id_token(cfg, oidc.discover(cfg), raw, nonce)


def reason(exc_info):
    return exc_info.value.detail + " | " + exc_info.value.public


# ---- settings ----

def test_nothing_set_means_oidc_is_off_and_a_partial_setup_is_reported():
    assert oidc.load_config({}) is None and oidc.load_config({"OIDC_ISSUER": "  "}) is None
    with pytest.raises(oidc.OidcError) as e:
        oidc.load_config({"OIDC_ISSUER": ISSUER, "OIDC_CLIENT_ID": CLIENT})
    assert "OIDC_CLIENT_SECRET" in e.value.detail and "OIDC_REDIRECT_URL" in e.value.detail


def test_settings_are_read_with_safe_defaults(monkeypatch):
    cfg = make_cfg(monkeypatch)
    assert cfg.scopes == ("openid", "email", "profile") and cfg.require_mfa is False and cfg.require_email_verified is True
    assert "x" * 12 not in repr(cfg)                                        # the client credential is never printed
    cfg = make_cfg(monkeypatch, OIDC_SCOPES="email, groups", OIDC_REQUIRE_MFA="true", OIDC_MFA_ACR_VALUES="urn:a, urn:b",
                   OIDC_EMAIL_CLAIM="preferred_username", OIDC_REQUIRE_EMAIL_VERIFIED="false")
    assert cfg.scopes == ("openid", "email", "groups") and cfg.require_mfa and cfg.mfa_acr == {"urn:a", "urn:b"}
    assert cfg.email_claim == "preferred_username" and cfg.require_email_verified is False


@pytest.mark.parametrize("issuer", ["http://idp.example.test", "ftp://idp.example.test", "https://user:pw@idp.example.test", "idp.example.test"])
def test_unsafe_issuer_addresses_are_refused(issuer):
    with pytest.raises(oidc.OidcError):
        oidc.load_config({"OIDC_ISSUER": issuer, "OIDC_CLIENT_ID": CLIENT, "OIDC_CLIENT_SECRET": "x" * 12, "OIDC_REDIRECT_URL": REDIRECT})


def test_plain_http_is_allowed_only_for_localhost_or_when_asked():
    base = {"OIDC_CLIENT_ID": CLIENT, "OIDC_CLIENT_SECRET": "x" * 12, "OIDC_REDIRECT_URL": "https://r.example.test/cb"}
    assert oidc.load_config(dict(base, OIDC_ISSUER="http://localhost:8080/realms/r"))
    assert oidc.load_config(dict(base, OIDC_ISSUER="http://keycloak.internal", OIDC_ALLOW_INSECURE_HTTP="true"))
    with pytest.raises(oidc.OidcError):
        oidc.load_config(dict(base, OIDC_ISSUER="http://keycloak.internal"))


# ---- discovery and keys ----

def test_discovery_is_cached_and_the_issuer_must_match(monkeypatch, idp):
    cfg = make_cfg(monkeypatch)
    oidc.discover(cfg)
    oidc.discover(cfg)
    assert len([c for c in idp.calls if c[1].endswith("openid-configuration")]) == 1
    oidc.reset_cache()
    idp.issuer_in_discovery = "https://other.example.test"
    with pytest.raises(oidc.OidcError) as e:
        oidc.discover(cfg)
    assert "issuer" in reason(e)


def test_redirects_and_bad_answers_from_the_provider_are_errors(monkeypatch, idp):
    cfg = make_cfg(monkeypatch)
    idp.redirect_on_discovery = True
    with pytest.raises(oidc.OidcError):
        oidc.discover(cfg)


def test_the_real_http_helpers_refuse_big_or_unreadable_answers():
    for response in (httpx.Response(200, content=b"x" * (oidc.MAX_BODY + 1)), httpx.Response(200, content=b"not json"),
                     httpx.Response(200, json=[1, 2]), httpx.Response(500, json={})):
        with pytest.raises(oidc.OidcError):
            oidc._read_json(response, "test")


# ---- the ID token ----

def test_a_good_token_gives_an_identity(monkeypatch, idp):
    cfg = make_cfg(monkeypatch)
    who = check(cfg, idp, idp.token())
    assert who == oidc.Identity(issuer=ISSUER, subject="user-123", email="person@example.test", name="A Person",
                                amr=("pwd", "otp"), acr="")


@pytest.mark.parametrize("change", [{"aud": "someone-else"}, {"iss": "https://other.example.test"}, {"exp": 1000, "iat": 900},
                                    {"iat": int(time.time()) + 3600, "exp": int(time.time()) + 7200}, {"nonce": "different-value-xyz"},
                                    {"nonce": None}, {"sub": None}, {"sub": ""}, {"sub": "s" * 256}, {"exp": None}])
def test_tokens_with_wrong_claims_are_refused(monkeypatch, idp, change):
    with pytest.raises(oidc.OidcError):
        check(make_cfg(monkeypatch), idp, idp.token(**change))


def test_several_audiences_need_a_matching_authorized_party(monkeypatch, idp):
    cfg = make_cfg(monkeypatch)
    with pytest.raises(oidc.OidcError) as e:
        check(cfg, idp, idp.token(aud=[CLIENT, "other"], azp="other"))
    assert "azp" in reason(e)
    assert check(cfg, idp, idp.token(aud=[CLIENT, "other"], azp=CLIENT)).subject == "user-123"


def test_a_token_signed_with_another_key_is_refused(monkeypatch, idp):
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    with pytest.raises(oidc.OidcError):
        check(make_cfg(monkeypatch), idp, idp.token(key=other))


def test_unsafe_algorithms_are_refused_before_any_key_is_used(monkeypatch, idp):
    cfg = make_cfg(monkeypatch)
    header = base64.urlsafe_b64encode(json.dumps({"alg": "none", "typ": "JWT", "kid": "k1"}).encode()).rstrip(b"=").decode()
    body = base64.urlsafe_b64encode(json.dumps({"iss": ISSUER, "aud": CLIENT, "sub": "u", "nonce": NONCE}).encode()).rstrip(b"=").decode()
    with pytest.raises(oidc.OidcError) as e:
        check(cfg, idp, header + "." + body + ".")
    assert "not allowed" in reason(e)
    public_pem = idp.jwks()["keys"][0]["n"]                                  # a classic attack: sign with HS256 using public key bytes
    forged = jwt.encode({"iss": ISSUER, "aud": CLIENT, "sub": "u", "nonce": NONCE, "iat": int(time.time()), "exp": int(time.time()) + 60},
                        public_pem, algorithm="HS256", headers={"kid": "k1"})
    with pytest.raises(oidc.OidcError) as e:
        check(cfg, idp, forged)
    assert "not allowed" in reason(e)


def test_garbage_and_huge_tokens_are_refused(monkeypatch, idp):
    cfg = make_cfg(monkeypatch)
    for raw in ("", "not-a-token", "a.b.c", "x" * (oidc.MAX_ID_TOKEN + 1), None, 12345):
        with pytest.raises(oidc.OidcError):
            check(cfg, idp, raw)


def test_ec_keys_work_too(monkeypatch, idp):
    idp.keys["ec1"] = (ec.generate_private_key(ec.SECP256R1()), "ES256")
    assert check(make_cfg(monkeypatch), idp, idp.token(kid="ec1")).subject == "user-123"


def test_a_new_signing_key_is_picked_up_once_but_not_on_every_unknown_key(monkeypatch, idp):
    cfg = make_cfg(monkeypatch)
    monkeypatch.setattr(oidc, "JWKS_REFETCH_MIN", 0)
    assert check(cfg, idp, idp.token()).subject == "user-123"
    idp.keys["k2"] = (rsa.generate_private_key(public_exponent=65537, key_size=2048), "RS256")      # the provider rotated its keys
    assert check(cfg, idp, idp.token(kid="k2")).subject == "user-123"
    before = len([c for c in idp.calls if c[1].endswith("/keys")])
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    idp.keys["unlisted"] = (other, "RS256")
    unknown = idp.token(kid="unlisted")
    del idp.keys["unlisted"]
    with pytest.raises(oidc.OidcError):
        check(cfg, idp, unknown)
    assert len([c for c in idp.calls if c[1].endswith("/keys")]) == before + 1       # one refetch for the unknown key
    monkeypatch.setattr(oidc, "JWKS_REFETCH_MIN", 3600)
    with pytest.raises(oidc.OidcError):
        check(cfg, idp, unknown)
    assert len([c for c in idp.calls if c[1].endswith("/keys")]) == before + 1       # and no second one within the minimum time


# ---- second factor and email ----

def test_the_second_factor_rule(monkeypatch, idp):
    cfg = make_cfg(monkeypatch, OIDC_REQUIRE_MFA="true", OIDC_MFA_ACR_VALUES="urn:strong")
    with pytest.raises(oidc.OidcError) as e:
        check(cfg, idp, idp.token(amr=["pwd"]))
    assert "second" in e.value.public
    with pytest.raises(oidc.OidcError):
        check(cfg, idp, idp.token(amr=None))
    for ok in ({"amr": ["pwd", "mfa"]}, {"amr": ["pwd", "otp"]}, {"amr": ["hwk", "pin"]}, {"amr": None, "acr": "urn:strong"}):
        assert check(cfg, idp, idp.token(**ok)).subject == "user-123"
    with pytest.raises(oidc.OidcError):
        check(cfg, idp, idp.token(amr=["pwd", "pwd"], acr="urn:weak"))                 # one factor twice is still one factor
    assert check(make_cfg(monkeypatch), idp, idp.token(amr=["pwd"])).subject == "user-123"      # not required: allowed


def test_the_email_rules(monkeypatch, idp):
    cfg = make_cfg(monkeypatch)
    for change in ({"email": None}, {"email": "no-at-sign"}, {"email": "a@" + "b" * 260}, {"email_verified": False}, {"email_verified": None},
                   {"email_verified": "no"}):
        with pytest.raises(oidc.OidcError):
            check(cfg, idp, idp.token(**change))
    assert check(cfg, idp, idp.token(email_verified="true")).email == "person@example.test"
    relaxed = make_cfg(monkeypatch, OIDC_REQUIRE_EMAIL_VERIFIED="false")
    assert check(relaxed, idp, idp.token(email_verified=None)).email == "person@example.test"
    named = make_cfg(monkeypatch, OIDC_EMAIL_CLAIM="preferred_username")
    assert check(named, idp, idp.token(preferred_username="Other@Example.Test")).email == "other@example.test"


# ---- sign-in steps ----

def test_pkce_matches_the_published_example():
    verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"                          # RFC 7636, appendix B
    assert oidc.pkce_challenge(verifier) == "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
    made = oidc.new_verifier()
    assert 43 <= len(made) <= 128 and oidc._VERIFIER.fullmatch(made) and made != oidc.new_verifier()


def test_state_comparison_is_exact():
    s = oidc.new_state()
    assert oidc.state_matches(s, s) and not oidc.state_matches(s, s + "x") and not oidc.state_matches(s, None)
    assert not oidc.state_matches("", "") and not oidc.state_matches(None, "x")


def test_the_sign_in_address_has_everything_and_no_credential(monkeypatch, idp):
    cfg = make_cfg(monkeypatch)
    url = oidc.authorization_url(cfg, oidc.discover(cfg), "st-1", "no-1", "chal-1")
    assert url.startswith(ISSUER + "/authorize?")
    query = dict(p.split("=", 1) for p in url.split("?", 1)[1].split("&"))
    assert query["response_type"] == "code" and query["client_id"] == CLIENT and query["state"] == "st-1" and query["nonce"] == "no-1"
    assert query["code_challenge"] == "chal-1" and query["code_challenge_method"] == "S256" and query["scope"] == "openid+email+profile"
    assert "https%3A%2F%2Fragleap.example.test%2Fauth%2Fcallback" in url and "x" * 12 not in url


def test_the_code_exchange_sends_the_verifier_and_returns_only_the_id_token(monkeypatch, idp):
    cfg = make_cfg(monkeypatch)
    verifier = oidc.new_verifier()
    assert oidc.exchange_code(cfg, oidc.discover(cfg), "one-time-code", verifier) == "x.y.z"
    form = idp.last_form
    assert form["grant_type"] == "authorization_code" and form["code"] == "one-time-code" and form["code_verifier"] == verifier
    assert form["client_id"] == CLIENT and form["client_secret"] == "x" * 12


def test_failed_exchanges_do_not_leak_the_code_or_the_credential(monkeypatch, idp):
    cfg = make_cfg(monkeypatch)
    disc = oidc.discover(cfg)
    verifier = oidc.new_verifier()
    idp.token_status = 400
    with pytest.raises(oidc.OidcError) as e:
        oidc.exchange_code(cfg, disc, "one-time-code", verifier)
    text = e.value.public + e.value.detail
    assert "one-time-code" not in text and "x" * 12 not in text
    idp.token_status, idp.token_body = 200, {"access_token": "only"}
    with pytest.raises(oidc.OidcError) as e:
        oidc.exchange_code(cfg, disc, "one-time-code", verifier)
    assert "id_token" in e.value.detail
    for code, ver in (("", verifier), ("c" * 3000, verifier), ("ok", "short"), ("ok", "bad verifier with spaces " * 3), (None, verifier)):
        with pytest.raises(oidc.OidcError):
            oidc.exchange_code(cfg, disc, code, ver)
