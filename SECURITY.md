# Security policy

MistralFit is experimental local tooling. Run a trusted llama.cpp binary, use model files from sources you trust, and keep the server bound to loopback unless you separately configure access controls.

For a suspected vulnerability, use the repository's private vulnerability-reporting option if enabled. If it is unavailable, open a redacted issue requesting a private reporting channel; omit exploit details, credentials, private prompts and model data until a channel is agreed.

Reports should describe the affected version, impact, prerequisites and a minimal safe reproduction. Automated tests cover malformed structural fixtures and HTML escaping; they are not a security audit.
