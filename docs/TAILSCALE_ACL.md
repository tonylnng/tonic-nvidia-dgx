# Tailscale Access — Hardening Recipe

Goal: expose the LiteLLM gateway on port 4000 over Tailscale only, allow SSH from admin devices only, and block everything else.

## 1. Spark side

```bash
# Bring Tailscale up with SSH and a tag
sudo tailscale up --ssh --advertise-tags=tag:dgx --hostname=spark

# Tailscale IP
tailscale ip -4
# e.g. 100.101.102.103

# Tailscale Serve — expose LiteLLM on HTTPS at spark.<tailnet>.ts.net
sudo tailscale serve --bg --https=443 http://127.0.0.1:4000
```

All Docker ports already bind to `127.0.0.1`, so the LiteLLM endpoint is reachable *only* through the tailnet.

## 2. Tailnet ACL (admin console)

```json
{
  "tagOwners": {
    "tag:dgx":   ["autogroup:admin"],
    "tag:agent": ["autogroup:admin"]
  },
  "acls": [
    // Admins can do anything on the Spark
    { "action": "accept",
      "src":   ["autogroup:admin"],
      "dst":   ["tag:dgx:22", "tag:dgx:443", "tag:dgx:4000"] },

    // Agent devices only get the LiteLLM gateway, nothing else
    { "action": "accept",
      "src":   ["tag:agent"],
      "dst":   ["tag:dgx:443"] }
  ],
  "ssh": [
    { "action": "check",
      "src":    ["autogroup:admin"],
      "dst":    ["tag:dgx"],
      "users":  ["autogroup:nonroot"] }
  ]
}
```

## 3. Agent side

```bash
sudo tailscale up --advertise-tags=tag:agent
curl https://spark.<tailnet>.ts.net/v1/models \
  -H "Authorization: Bearer sk-virtualkey-of-this-agent"
```

## 4. Audit

```bash
# Who connected today
tailscale status --json | jq '.Peer[] | {Name,TailscaleIPs,LastSeen}'
```
