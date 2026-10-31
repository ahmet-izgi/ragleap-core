"""OpenID Connect sign-in: the authorization code flow with PKCE and strict ID token checks.

This module has no database and no routes. It holds the parts that must be exactly right and testable without a network:
reading the settings, finding the provider's endpoints and signing keys, building the sign-in address, exchanging the code
for an ID token, and validating that token (signature, issuer, audience, expiry, nonce, safe algorithms only, and an optional
second-factor rule). Everything is off unless the OIDC_* settings are present. The provider performs the actual sign-in and
the second factor; RagLeap only checks what the provider says."""
import base64
import hashlib
import hmac
import logging
import os
import re
import secrets
import threading
import time
import urllib.parse
from dataclasses import dataclass, field

import httpx
import jwt

logger = logging.getLogger(__name__)

ALLOWED_ALGS = ("RS256", "RS384", "RS512", "ES256", "ES384", "ES512", "PS256", "PS384", "PS512")
# Authentication method values (RFC 8176) that stand for one factor each. Two different ones count as a second factor.
FACTOR_AMR = frozenset({"pwd", "otp", "hwk", "swk", "sms", "tel", "fpt", "face", "iris", "retina", "vbm", "pin", "pop", "sc", "kba"})
MAX_BODY = 1_000_000
MAX_ID_TOKEN = 16_384
LEEWAY = 60
CACHE_SECONDS = 3600
JWKS_REFETCH_MIN = 60
_VERIFIER = re.compile(r"[A-Za-z0-9._~-]{43,128}")
_LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")

_lock = threading.Lock()
_cache = {"disc": {}, "jwks": {}}          # key -> (valid until, value)
_jwks_fetched = {}                         # issuer -> time of the last fetch


class OidcError(Exception):
    """`public` is safe to show to the person signing in; `detail` is for the log and never holds tokens or secrets."""

    def __init__(self, public, detail=""):
        super().__init__(public)
        self.public = public
        self.detail = detail


@dataclass(frozen=True)
class Config:
    issuer: str
    client_id: str
    client_secret: str = field(repr=False)
    redirect_url: str
    scopes: tuple = ("openid", "email", "profile")
    require_mfa: bool = False
    mfa_acr: frozenset = frozenset()
    email_claim: str = "email"
    require_email_verified: bool = True
    allow_http: bool = False


@dataclass(frozen=True)
class Identity:
    issuer: str
    subject: str
    email: str
    name: str = ""
    amr: tuple = ()
    acr: str = ""


def _truthy(value, default=False):
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _check_url(url, allow_http, what):
    parts = urllib.parse.urlsplit(url or "")
    if not parts.netloc or parts.username or parts.password:
        raise OidcError("OIDC is not set up correctly.", "%s is not a plain URL" % what)
    local = (parts.hostname or "").lower() in _LOCAL_HOSTS
    if parts.scheme != "https" and not (parts.scheme == "http" and (allow_http or local)):
        raise OidcError("OIDC is not set up correctly.", "%s must use https (http only for localhost)" % what)
    return url


def load_config(env=None):
    """None when OIDC is not configured at all. Raises OidcError when it is configured only in part or wrongly."""
    env = os.environ if env is None else env
    names = ("OIDC_ISSUER", "OIDC_CLIENT_ID", "OIDC_CLIENT_SECRET", "OIDC_REDIRECT_URL")
    present = [n for n in names if (env.get(n) or "").strip()]
    if not present:
        return None
    missing = [n for n in names if n not in present]
    if missing:
        raise OidcError("OIDC is partly set up.", "missing: " + ", ".join(missing))
    allow_http = _truthy(env.get("OIDC_ALLOW_INSECURE_HTTP"))
    issuer = env["OIDC_ISSUER"].strip()
    redirect = env["OIDC_REDIRECT_URL"].strip()
    _check_url(issuer, allow_http, "OIDC_ISSUER")
    _check_url(redirect, allow_http, "OIDC_REDIRECT_URL")
    scopes = [s for s in re.split(r"[\s,]+", env.get("OIDC_SCOPES") or "openid email profile") if s]
    if "openid" not in scopes:
        scopes.insert(0, "openid")
    return Config(issuer=issuer, client_id=env["OIDC_CLIENT_ID"].strip(), client_secret=env["OIDC_CLIENT_SECRET"].strip(),
                  redirect_url=redirect, scopes=tuple(scopes), require_mfa=_truthy(env.get("OIDC_REQUIRE_MFA")),
                  mfa_acr=frozenset(v.strip() for v in (env.get("OIDC_MFA_ACR_VALUES") or "").split(",") if v.strip()),
                  email_claim=(env.get("OIDC_EMAIL_CLAIM") or "email").strip(),
                  require_email_verified=_truthy(env.get("OIDC_REQUIRE_EMAIL_VERIFIED"), True), allow_http=allow_http)


def reset_cache():
    with _lock:
        _cache["disc"].clear()
        _cache["jwks"].clear()
        _jwks_fetched.clear()


# ---- network (replaced in tests) ----

def _make_client():
    return httpx.Client(timeout=10.0, follow_redirects=False, headers={"Accept": "application/json"})


def _read_json(response, what):
    if response.status_code != 200:
        raise OidcError("The sign-in provider did not accept the request.", "%s answered %s" % (what, response.status_code))
    if len(response.content) > MAX_BODY:
        raise OidcError("The sign-in provider sent too much data.", "%s answer too large" % what)
    try:
        data = response.json()
    except ValueError:
        raise OidcError("The sign-in provider sent an answer that cannot be read.", "%s answer is not JSON" % what)
    if not isinstance(data, dict):
        raise OidcError("The sign-in provider sent an answer that cannot be read.", "%s answer is not an object" % what)
    return data


def _http_get_json(url, what):
    try:
        with _make_client() as client:
            return _read_json(client.get(url), what)
    except httpx.HTTPError as e:
        raise OidcError("The sign-in provider cannot be reached.", "%s: %s" % (what, type(e).__name__))


def _http_post_form(url, data, what):
    try:
        with _make_client() as client:
            return _read_json(client.post(url, data=data), what)
    except httpx.HTTPError as e:
        raise OidcError("The sign-in provider cannot be reached.", "%s: %s" % (what, type(e).__name__))


# ---- provider endpoints and keys ----

def _cached(kind, key, loader):
    now = time.monotonic()
    with _lock:
        hit = _cache[kind].get(key)
        if hit and hit[0] > now:
            return hit[1]
    value = loader()
    with _lock:
        _cache[kind][key] = (time.monotonic() + CACHE_SECONDS, value)
    return value


def discover(cfg):
    def load():
        doc = _http_get_json(cfg.issuer.rstrip("/") + "/.well-known/openid-configuration", "discovery")
        if doc.get("issuer") != cfg.issuer:
            raise OidcError("The sign-in provider does not match the configured issuer.", "issuer in discovery differs")
        out = {}
        for key in ("authorization_endpoint", "token_endpoint", "jwks_uri"):
            out[key] = _check_url(doc.get(key), cfg.allow_http, key)
        return out
    return _cached("disc", cfg.issuer, load)


def _jwks(cfg, disc, force=False):
    if force:
        with _lock:
            last = _jwks_fetched.get(cfg.issuer, 0.0)
            if time.monotonic() - last < JWKS_REFETCH_MIN:
                raise OidcError("The signing key of this token is not known.", "key set was fetched less than a minute ago")
            _cache["jwks"].pop(cfg.issuer, None)

    def load():
        data = _http_get_json(disc["jwks_uri"], "key set")
        keys = data.get("keys")
        if not isinstance(keys, list) or not keys:
            raise OidcError("The sign-in provider has no signing keys.", "key set is empty")
        with _lock:
            _jwks_fetched[cfg.issuer] = time.monotonic()
        return keys
    return _cached("jwks", cfg.issuer, load)


def _find_key(keys, kid, alg):
    candidates = [k for k in keys if isinstance(k, dict) and k.get("use", "sig") == "sig"
                  and (kid is None or k.get("kid") == kid) and k.get("alg", alg) == alg]
    if kid is None and len(candidates) != 1:
        return None
    return candidates[0] if candidates else None


# ---- sign-in steps ----

def new_state():
    return secrets.token_urlsafe(32)


def new_verifier():
    return secrets.token_urlsafe(64)[:96]


def pkce_challenge(verifier):
    return base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode("ascii")


def state_matches(expected, got):
    return bool(expected) and isinstance(got, str) and hmac.compare_digest(expected.encode("utf-8"), got.encode("utf-8"))


def authorization_url(cfg, disc, state, nonce, challenge):
    query = urllib.parse.urlencode({
        "response_type": "code", "client_id": cfg.client_id, "redirect_uri": cfg.redirect_url, "scope": " ".join(cfg.scopes),
        "state": state, "nonce": nonce, "code_challenge": challenge, "code_challenge_method": "S256"})
    sep = "&" if "?" in disc["authorization_endpoint"] else "?"
    return disc["authorization_endpoint"] + sep + query


def exchange_code(cfg, disc, code, verifier):
    """Trades the one-time code for an ID token. Returns the raw ID token text."""
    if not isinstance(code, str) or not 1 <= len(code) <= 2048:
        raise OidcError("The sign-in response is not valid.", "code missing or too long")
    if not isinstance(verifier, str) or not _VERIFIER.fullmatch(verifier):
        raise OidcError("The sign-in response is not valid.", "bad code verifier")
    data = _http_post_form(disc["token_endpoint"], {
        "grant_type": "authorization_code", "code": code, "redirect_uri": cfg.redirect_url, "client_id": cfg.client_id,
        "client_secret": cfg.client_secret, "code_verifier": verifier}, "token request")
    raw = data.get("id_token")
    if not isinstance(raw, str) or not raw:
        raise OidcError("The sign-in provider did not send an ID token.", "no id_token in the token answer")
    return raw


def check_mfa(cfg, claims):
    if not cfg.require_mfa:
        return
    amr = claims.get("amr")
    methods = {m for m in amr if isinstance(m, str)} if isinstance(amr, list) else set()
    if "mfa" in methods or len(methods & FACTOR_AMR) >= 2 or (cfg.mfa_acr and claims.get("acr") in cfg.mfa_acr):
        return
    raise OidcError("A second sign-in factor is required for this system.", "token has no multi-factor evidence (amr/acr)")


def validate_id_token(cfg, disc, raw, nonce):
    """Returns an Identity, or raises OidcError. Never trusts anything in the token before its signature is checked."""
    if not isinstance(raw, str) or not raw or len(raw) > MAX_ID_TOKEN:
        raise OidcError("The sign-in response is not valid.", "ID token missing or too large")
    try:
        header = jwt.get_unverified_header(raw)
    except jwt.PyJWTError:
        raise OidcError("The sign-in response is not valid.", "ID token header cannot be read")
    alg, kid = header.get("alg"), header.get("kid")
    if alg not in ALLOWED_ALGS:
        raise OidcError("The sign-in response is not valid.", "algorithm %r is not allowed" % (alg,))
    keys = _jwks(cfg, disc)
    jwk = _find_key(keys, kid, alg)
    if jwk is None and kid is not None:
        jwk = _find_key(_jwks(cfg, disc, force=True), kid, alg)
    if jwk is None:
        raise OidcError("The signing key of this token is not known.", "no key for kid %r" % (kid,))
    try:
        claims = jwt.decode(raw, jwt.PyJWK(jwk).key, algorithms=[alg], audience=cfg.client_id, issuer=cfg.issuer, leeway=LEEWAY,
                            options={"require": ["exp", "iat", "iss", "aud", "sub"]})
    except jwt.PyJWTError as e:
        raise OidcError("The sign-in response is not valid.", "ID token rejected: " + type(e).__name__)
    aud = claims.get("aud")
    if isinstance(aud, list) and len(aud) > 1 and claims.get("azp") != cfg.client_id:
        raise OidcError("The sign-in response is not valid.", "azp does not match for a token with several audiences")
    if not state_matches(nonce, claims.get("nonce")):
        raise OidcError("The sign-in response is not valid.", "nonce does not match")
    subject = claims.get("sub")
    if not isinstance(subject, str) or not 1 <= len(subject) <= 255:
        raise OidcError("The sign-in response is not valid.", "subject missing or too long")
    check_mfa(cfg, claims)
    email = claims.get(cfg.email_claim)
    if not isinstance(email, str) or "@" not in email or len(email) > 254:
        raise OidcError("The provider did not send an email address for you.", "claim %r missing" % cfg.email_claim)
    if cfg.require_email_verified and claims.get("email_verified") not in (True, "true"):
        raise OidcError("Your email address is not verified by the provider.", "email_verified is not true")
    name = claims.get("name")
    amr = claims.get("amr")
    return Identity(issuer=cfg.issuer, subject=subject, email=email.strip().lower(), name=name[:200] if isinstance(name, str) else "",
                    amr=tuple(m for m in amr if isinstance(m, str)) if isinstance(amr, list) else (),
                    acr=claims.get("acr") if isinstance(claims.get("acr"), str) else "")
