"""XPU collector — endpoint routing + dispatch + result aggregation.

Business-agnostic: takes pre-resolved ExecutionSpecs, handles endpoint
selection and dispatch. Does NOT import test_spec.
"""
import threading

from ttk.remote import DATA, PERF
from ttk.remote.dispatcher import RemoteExecutionError, dispatch_to_remote


def _select_run_specs(specs, xpu_mode):
    """Choose which specs to run + identify priority.

    Availability filtering moved upstream into EndpointView.resolve_providers
    (profiling._do_xpu_profiling); all incoming specs are already resolvable,
    so no has_endpoint predicate is needed here.

    priority = first spec in input order.
    Mode dispatch:
        DATA       -> [priority] or []   (save non-priority output transfer)
        PERF       -> all
        DATA|PERF  -> all
    """
    priority = specs[0] if specs else None
    if xpu_mode == DATA:
        run = [priority] if priority else []
    else:  # PERF or DATA|PERF -> all
        run = list(specs)
    return run, priority


def _per_spec_mode(spec, priority, xpu_mode):
    """DATA|PERF: priority gets DATA|PERF, others get PERF (save non-priority
    output transfer). When priority is None (no available endpoint), every
    dispatched spec gets PERF only — no provider yields Data, but the run
    still completes for PERF (caller records accuracy-skipped via status)."""
    if xpu_mode == (DATA | PERF):
        return (DATA | PERF) if (priority and spec.provider == priority.provider) else PERF
    return xpu_mode


def collect_xpu_results(specs, *, inputs, input_names, mode,
                        tenant_id, op_name="", op_type=None, attrs=None,
                        input_formats=None,
                        spec_search_roots=None, tmp_root=None, runtime: int = 3,
                        param_order=None):
    """Dispatch specs to xpu-server, return aggregated results.

    Args:
        specs: [ExecutionSpec, ...] — pre-resolved execution plans
        inputs: numpy input arrays
        input_names: input parameter names
        mode: DATA / PERF / DATA|PERF
        tenant_id: tenant ID
        op_name: operator name (for dispatch)
        op_type: operator type (for dispatch)
        attrs: operator attributes dict
        spec_search_roots: spec search paths (for 424 retry)

    Returns:
        {provider: {"status", "api", "outputs"?, "perf"?, "error"?}}
    """
    # Results are keyed by provider -> one spec per provider. Dedup preserving
    # priority order (first wins); guards the public seam against future callers
    # (ACLNN/E2E) that might build specs with a repeated provider.
    _seen = set()
    specs = [s for s in specs if not (s.provider in _seen or _seen.add(s.provider))]

    from ttk.remote.endpoint_view import EndpointView
    ev = EndpointView()                       # per-process Singleton; the ONLY endpoint decision point
    run_specs, priority = _select_run_specs(specs, mode)
    results = {}

    def _dispatch_one(spec, spec_mode):
        provider = spec.provider
        api_label = spec.api or "custom"   # pre-computed fallback (used on error paths)
        ep = ev.pick_endpoint(spec.provider)   # was: _pick_endpoint(provider, endpoints)
        if ep is None:
            return provider, {
                "status": "FAIL", "api": api_label,
                "error": f"no alive endpoint supports {provider}",
            }
        try:
            r = dispatch_to_remote(
                op_name=op_name, op_type=op_type,
                inputs=inputs, input_names=input_names,
                provider=provider,
                attrs=attrs or {},
                input_formats=input_formats,
                endpoint_host=ep.host, endpoint_port=ep.port,
                tenant_id=tenant_id, mode=spec_mode, return_result=True,
                execution_type=spec.type, api=spec.api,
                spec_module=spec.spec_module,
                spec_class=spec.spec_class,
                spec_file=spec.spec_file,
                spec_search_roots=spec_search_roots,
                tmp_root=tmp_root,
                runtime=runtime,
                param_order=param_order,
            )
            # Success: server-resolved API (r.api) wins over the client-side guess.
            resolved_api = (r.api if r else None) or api_label
            return provider, {
                "status": "PASS", "api": resolved_api,
                "outputs": r.outputs, "perf": r.perf,
            }
        except RemoteExecutionError as e:
            return provider, {
                "status": "FAIL", "api": api_label, "error": str(e),
            }
        except Exception as e:
            return provider, {
                "status": "FAIL", "api": api_label,
                "error": f"{type(e).__name__}: {e}",
            }

    # Specs not run simply do not appear in results — availability is decided
    # upstream by EndpointView.resolve_providers (contract).
    if len(run_specs) == 1:
        p, r = _dispatch_one(run_specs[0], _per_spec_mode(run_specs[0], priority, mode))
        results[p] = r
    elif len(run_specs) > 1:
        threads = []
        for s in run_specs:
            t = threading.Thread(target=lambda s=s: results.update(
                [_dispatch_one(s, _per_spec_mode(s, priority, mode))]))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

    return results
