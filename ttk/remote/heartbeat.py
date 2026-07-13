"""
Tenant heartbeat subprocess.

Runs as independent subprocess. Probes all endpoints concurrently via
GET /v1/heartbeat (merged health + detect + register endpoint), writing the
aggregated health file via atomic_write_json. Detects parent process death
and sends DELETE cleanup.

TLS is delegated to the shared ttk.remote.tls module (ca or cert+key -> HTTPS).
"""
import json
import logging
import os
import ssl
import threading
import time

from ttk.remote.tls import build_tls_connection

HEARTBEAT_INTERVAL_S = 11
HEARTBEAT_TIMEOUT_S = 5

log = logging.getLogger(__name__)


def heartbeat_loop(endpoints, tenant_id, health_path, tls=None):
    """Heartbeat subprocess entry point.

    Concurrently probes all endpoints via GET /v1/heartbeat (merged endpoint:
    health + detect + register in one round-trip). Writes per-endpoint health
    state to health_path via atomic_write_json after each cycle.

    tls: Optional[dict] = {"ca_cert","cert","key"}; 委托 tls.build_tls_connection
    （ca 或 cert+key 成对 -> HTTPS）；None / {} -> plain HTTP.

    Parent death detection: os.getppid() changes (reparented to init or a
    subreaper) when the parent dies. We compare against the ppid captured at
    start, not ==1, so this is robust under non-init subreapers (systemd,
    containers). Triggers DELETE /v1/tenant/xxx to all endpoints then exits.

    Args:
        endpoints: list of Endpoint objects (with .host, .port attributes)
        tenant_id: TTK tenant ID string
        health_path: absolute path to health state JSON file
        tls: optional dict from tls.tls_from_config (ca_cert/cert/key)
    """
    from .health_file import atomic_write_json
    log.info("HB started: tenant=%s health_path=%s n_endpoints=%d tls=%s",
             tenant_id, health_path, len(endpoints), bool(tls))
    original_ppid = os.getppid()  # parent at fork time; reparenting on parent
                                  # death (to init OR a subreaper) is what we detect.
    while True:
        # Detect parent death: ppid changes when the process is reparented.
        # Comparing to the ORIGINAL parent (not ==1) is robust across non-init
        # subreapers (systemd, container runtimes) — the old ==1 check missed
        # those. HB is forked (multiprocessing.Process, default ctx), so its
        # parent is the TTK process itself.
        if os.getppid() != original_ppid:
            log.info("Parent died (ppid %d->%d), cleaning up tenant %s",
                     original_ppid, os.getppid(), tenant_id)
            _cleanup_all(endpoints, tenant_id, tls)
            return

        results = {}
        threads = []
        for ep in endpoints:
            # Per-thread holder: each probe writes only to its own dict, so there
            # is no shared mutable state across threads (no lock needed, and no
            # latent risk if a future change adds read-modify-write).
            res = {}
            t = threading.Thread(target=_probe_one,
                                 args=(ep, tenant_id, res, tls), daemon=True)
            t.start()
            threads.append((t, ep, res))
        for t, ep, res in threads:
            ep_key = f"{ep.host}:{ep.port}"
            t.join(timeout=HEARTBEAT_TIMEOUT_S)
            if res:                    # thread wrote its result
                results.update(res)
            else:                      # timed out -> mark dead
                results[ep_key] = {
                    "alive": False, "last_seen": None,
                    "providers": [], "hardware": "", "ts": time.time(),
                }

        atomic_write_json(health_path, {"endpoints": results})
        time.sleep(HEARTBEAT_INTERVAL_S)


def _probe_one(endpoint, tenant_id, out_dict, tls):
    """Probe one endpoint via GET /v1/heartbeat; merge health+detect+register."""
    ep_key = f"{endpoint.host}:{endpoint.port}"
    conn = None
    try:
        conn = build_tls_connection(endpoint.host, endpoint.port,
                                    HEARTBEAT_TIMEOUT_S, tls)
        conn.request("GET", f"/v1/heartbeat?tenant_id={tenant_id}")
        resp = conn.getresponse()
        raw_body = resp.read()
        alive = 200 <= resp.status < 300
        if alive:
            try:
                body = json.loads(raw_body)
            except (json.JSONDecodeError, ValueError):
                log.warning("probe %s: non-JSON response (status %s), marking dead",
                            ep_key, resp.status)
                out_dict[ep_key] = {
                    "alive": False, "last_seen": None,
                    "providers": [], "hardware": "", "ts": time.time(),
                }
                return
        else:
            body = {}
        out_dict[ep_key] = {
            "alive": alive,
            "last_seen": time.time() if alive else None,
            "providers": body.get("providers", []),
            "hardware": body.get("hardware", ""),
            "ts": time.time(),
        }
    except Exception as e:
        # TLS cert/handshake failures (IP-SAN mismatch, expired, untrusted CA) are
        # config errors that won't self-heal → ERROR; transient (refused/timeout) → debug.
        if isinstance(e, ssl.SSLError):
            log.error("probe %s TLS handshake/cert failed "
                      "(check cert IP/SAN matches endpoint.host): %s", ep_key, e)
        else:
            log.debug("probe %s failed: %s", ep_key, e)
        out_dict[ep_key] = {
            "alive": False, "last_seen": None,
            "providers": [], "hardware": "", "ts": time.time(),
        }
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def _cleanup_all(endpoints, tenant_id, tls):
    """Send DELETE cleanup to all endpoints (threaded with tls)."""
    for ep in endpoints:
        try:
            conn = build_tls_connection(ep.host, ep.port,
                                        HEARTBEAT_TIMEOUT_S, tls)
            conn.request("DELETE", f"/v1/tenant/{tenant_id}")
            resp = conn.getresponse()
            resp.read()
            conn.close()
            log.info("Cleaned tenant %s on %s:%s", tenant_id, ep.host, ep.port)
        except Exception as e:
            if isinstance(e, ssl.SSLError):
                log.error("cleanup %s:%s TLS handshake/cert failed: %s", ep.host, ep.port, e)
            else:
                log.debug("Cleanup %s:%s failed: %s", ep.host, ep.port, e)
