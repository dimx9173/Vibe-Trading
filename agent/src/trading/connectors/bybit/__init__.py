"""Bybit ccxt connector (demo + live) with host-separated paper guard.

Transport is ccxt's ``bybit`` class. The paper guard is host separation
(``paper_guard="host_separated"``), backed by the following bring-up facts:

- Structural runtime discriminator: the REST subdomain plus Bybit's demo key
  namespace. ``paper`` profiles resolve to ``api-demo.bybit.com`` through
  ``enable_demo_trading(True)``; live profiles resolve to ``api.bybit.com``.
  A Bybit demo account is a separate key namespace with its own user id, so
  demo keys cannot authenticate against the live host and live keys cannot
  reach the demo host. ``api-testnet.bybit.com`` is a DIFFERENT system
  (separate keys, separate ledger) that this connector never selects:
  ``set_sandbox_mode`` is never called because it switches ccxt to the
  testnet URLs where demo keys are meaningless.
- Declared surface limits: ``market_type`` is ``spot`` (``BASE/QUOTE``) or
  ``linear`` (``BASE/QUOTE:SETTLE`` USDT/USDC-settled perpetual) only.
  Inverse contracts, options and portfolio margin are not implemented. No
  private websocket streaming is offered: Bybit's demo environment does not
  support WS trade, so a single host-separated REST client is the whole
  transport.
- Verification status: the hermetic tests in ``agent/tests/test_bybit_connector.py``
  and ``agent/tests/test_bybit_profiles.py`` cover config validation, the
  host guard, symbol/interval shaping, read shaping and the fail-closed write
  matrix against fake ccxt exchanges (no network). Signed reads and signed
  order placement have NOT been exercised against the broker from this
  workspace; only the contract shapes are pinned offline. Runtime evidence
  for the read/write paths must come from a demo key connected through the
  service layer before any live trading claim.
"""
