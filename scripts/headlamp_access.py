#!/usr/bin/env python3
"""Admin-only provisioning: individual, expiring lab UI tokens; never print tokens."""
import argparse, base64, hashlib, json, os, re, subprocess
from datetime import datetime, timezone
from pathlib import Path
NS = "systems-world-model-lab"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "HostName=100.98.114.55", "-o", "HostKeyAlias=amd-halo", "rajnishtala@amd-halo"]
K = "kubectl --kubeconfig=$HOME/.kube/graphrobin.yaml -n " + NS + " "
def remote(command, data=None):
    p = subprocess.run(SSH + [K + command], input=data, text=True, capture_output=True, timeout=45)
    if p.returncode:
        raise RuntimeError(p.stderr[-2000:])
    return p.stdout.strip()
def account_name(identity):
    slug = re.sub(r"[^a-z0-9]+", "-", identity.lower().split("@")[0]).strip("-")[:28] or "user"
    return "swm-ui-" + slug + "-" + hashlib.sha256(identity.lower().encode()).hexdigest()[:8]
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("identity", help="Person's email; label only, not verified email/SSO authentication")
    p.add_argument("--role", choices=["viewer", "operator"], default="viewer")
    p.add_argument("--revoke", action="store_true")
    p.add_argument("--output", type=Path, help="New private token file (must not exist)")
    a = p.parse_args(); name = account_name(a.identity)
    if a.revoke:
        remote("delete rolebinding " + name + " --ignore-not-found=true")
        remote("delete serviceaccount " + name + " --ignore-not-found=true")
        print("Revoked " + name); return
    if not a.output:
        p.error("--output is required; tokens are never printed")
    # Reserve a private file before making changes. Refuse symlinks/existing files.
    fd = os.open(a.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        meta = {"name": name, "namespace": NS, "labels": {"app.kubernetes.io/part-of": "systems-world-model", "app.kubernetes.io/component": "lab-user"}, "annotations": {"systems-world-model/identity": a.identity}}
        sa = {"apiVersion": "v1", "kind": "ServiceAccount", "metadata": meta, "automountServiceAccountToken": False}
        binding = {"apiVersion": "rbac.authorization.k8s.io/v1", "kind": "RoleBinding", "metadata": meta, "roleRef": {"apiGroup": "rbac.authorization.k8s.io", "kind": "Role", "name": "swm-ui-" + a.role}, "subjects": [{"kind": "ServiceAccount", "name": name, "namespace": NS}]}
        # RoleRefs are immutable: role changes require explicit revocation first.
        remote("apply -f -", json.dumps({"apiVersion": "v1", "kind": "List", "items": [sa, binding]}))
        token = remote("create token " + name + " --duration=8h")
        claims = json.loads(base64.urlsafe_b64decode(token.split(".")[1] + "=="))
        with os.fdopen(fd, "w") as f:
            f.write(token + "\n")
        fd = None
        print(json.dumps({"account": name, "role": a.role, "expires_utc": datetime.fromtimestamp(claims["exp"], timezone.utc).isoformat(), "token_file": str(a.output.resolve())}, indent=2))
    finally:
        if fd is not None:
            os.close(fd)
            a.output.unlink(missing_ok=True)
if __name__ == "__main__":
    main()
