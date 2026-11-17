# Private lab resource UI

Headlamp v0.46.0 is deployed only in `systems-world-model-lab`. The image is pinned
by digest in `deploy/headlamp.json`. The five existing services are unchanged.

## Open and sign in

Connect to the existing Tailscale network, then open:

http://amd-halo.tailf2f933.ts.net:8083/

This listener is tailnet-only, not public. HTTP travels inside Tailscale's encrypted
tunnel; it is **not browser HTTPS**. Tailnet HTTPS is not enabled, and we did not
change that organization setting. Existing port 8080/8082 routes are preserved.
Users also need network access permitted by the existing tailnet policy; Kubernetes
RBAC alone does not enroll them in Tailscale or grant a tailnet device access.

Paste your individual token into Headlamp. The initial operator account is labelled
`099.raj@gmail.com`. This is **token authentication, not Google/email SSO**; the
email is an administrative identity label, not an identity-provider assertion.
Tokens expire after eight hours and must not be shared. SSO is a future option
requiring the cluster API server and Headlamp to trust the same identity provider.

On a new browser, use **Cluster settings** to set both **Default namespace** and
**Allowed namespaces** to `systems-world-model-lab` (press Enter to add the latter).
These settings simplify navigation; API-server RBAC enforces the actual boundary.
Cluster-wide pages may report Forbidden. Node shell and pod debug are not granted.
The Codex browser used during setup already has these navigation settings.

## Roles and boundaries

- Viewer: read lab workload state, pod logs, events, services, PVC and quota status.
- Operator: viewer permissions plus patch/scale the five existing service Deployments,
  create/update/delete experiment Jobs, and delete pods. Deployment names are
  explicitly restricted; creating new Deployments is not included.
- Neither role grants access to other namespaces or cluster resources, direct Secret
  access, RBAC changes, token creation, network-policy/quota changes, or PVC deletion.
- Operators can run code through Jobs and mount namespace data or service accounts;
  they must be trusted with the **whole lab's data**, including the results PVC.
  Direct Secret/PVC API restrictions do not prevent a workload reading or modifying
  mounted data. Keep higher-privilege service accounts and sensitive secrets out of
  this namespace. This is a team lab, not isolation between mutually untrusted users.
- The existing restricted Pod Security Admission and default-deny policies remain.
  The UI has a token-free kubeconfig, no mounted service-account token, and no RBAC
  binding. Browser requests use the individual user's token. Only Headlamp gets
  API-server egress; other policies are unchanged.

## Add, renew, or revoke access

Run these from the repo with the existing admin SSH access. They do not email users.
The helper writes a mode-0600 token file and never prints its contents. Files under
`artifacts/` are git-ignored. Use a new output filename for each issuance.

```sh
mkdir -p artifacts/headlamp
python3 scripts/headlamp_access.py teammate@example.com --role viewer \
  --output artifacts/headlamp/teammate-token.txt
# Use --role operator for trusted experiment operators.
# Re-run with the same identity and role, using a new output file, to renew.
python3 scripts/headlamp_access.py teammate@example.com --revoke
```

Revocation deletes the RoleBinding and ServiceAccount. To change a role, revoke
first and issue again; RoleBinding roleRef is immutable. Deleting/recreating the
ServiceAccount also invalidates tokens belonging to the old account UID. Reissuing
without revocation does not invalidate previously issued tokens before their expiry.
Deliver each person's token privately; never send an administrator kubeconfig.

## Deploy or remove

On the host, use the existing kubeconfig and apply only this namespace manifest:

```sh
kubectl --kubeconfig=$HOME/.kube/graphrobin.yaml -n systems-world-model-lab \
  apply -f deploy/headlamp.json
kubectl --kubeconfig=$HOME/.kube/graphrobin.yaml -n systems-world-model-lab \
  rollout status deployment/swm-headlamp
```

After deployment, obtain the ClusterIP of `service/swm-headlamp`. The current one is
`10.43.80.214`. Tailscale Serve on the host persists this private proxy:

```sh
tailscale serve --bg --http=8083 http://10.43.80.214:4466
```

If the Service is deleted/recreated, update that proxy to its new ClusterIP. The
network policy is pinned to this lab's API endpoint and host addresses; inspect and
update it deliberately when moving hosts. No NodePort, public ingress, shared
administrator token, ClusterRole, or ClusterRoleBinding was installed.

To remove the UI: disable only `tailscale serve --http=8083 off`, revoke issued
accounts, then `kubectl ... delete -f deploy/headlamp.json`. Do not reset all Serve
routes or delete the namespace/PVC. Revoke accounts individually with the helper.

## Capacity and verification

The 4-CPU/6-GiB quota is unchanged. Headlamp is capped at 50m CPU/256Mi. The staging
pod now uses 50m CPU instead of 100m: together they occupy the existing 100m headroom.
Five services + UI + staging fit at 4 CPU/6 GiB and seven pods. Two-CPU training still
requires pausing inference, and the UI stays available while training runs. Updated
runners copy the current staging manifest and check available CPU instead of waiting
for the obsolete exact baseline of 1600m. Do not run concurrent experiments.

`reports/evidence/headlamp-access/checks.json` records actual operator-token requests
through Headlamp, including cross-namespace denial, forbidden privilege changes,
and server-dry-run deployment/Job operations. To repeat with a current operator token:

```sh
python3 scripts/verify_headlamp_access.py --token-file artifacts/headlamp/raj-token.txt
```

The browser login page and namespace settings were verified. Authenticated API
operations were verified independently; the user's token was not pasted into an
automated browser session. The viewer role is also checked separately in the evidence.
