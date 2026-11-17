/* Lab-only integration for Headlamp v0.46.0 behind ngrok OAuth. */
(function () {
  const { React, ReactRouter, registerAppBarAction } = window.pluginLib;
  const key = 'swm.logout-generation';
  const readGeneration = () => { try { return localStorage.getItem(key); } catch (_) { return null; } };
  const initialGeneration = readGeneration();
  const channel = typeof BroadcastChannel === 'function' ? new BroadcastChannel('swm-session') : null;
  let subscribed = false;
  let leaving = false;
  let checking = false;
  async function revalidate() {
    if (leaving || checking) return;
    checking = true;
    document.documentElement.style.visibility = 'hidden';
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 10000);
    try {
      // History can restore the app shell even after the OAuth cookie is gone.
      // Do not follow an IdP redirect inside fetch; start a top-level login instead.
      const response = await fetch('/?swm_session_check=' + Date.now(), {
        credentials: 'same-origin', cache: 'no-store', redirect: 'error',
        signal: controller.signal
      });
      if (!response.ok) throw new Error('Session check failed');
      if (!leaving) document.documentElement.style.visibility = '';
    } catch (_) {
      if (!leaving) {
        leaving = true;
        window.location.replace('/ngrok/login?redirect_path=%2Fc%2Fsystems-world-model-lab%2Fworkloads');
      }
    } finally {
      clearTimeout(timeout);
      checking = false;
    }
  }
  function leave() {
    if (leaving) return;
    leaving = true;
    // Hide already-rendered data immediately, then unload watches and connections.
    document.documentElement.style.visibility = 'hidden';
    window.location.replace('/ngrok/logout');
  }
  function logout() {
    if (leaving) return;
    try { localStorage.setItem(key, Date.now() + ':' + Math.random()); } catch (_) {}
    if (channel) channel.postMessage('logout');
    leave();
  }
  function checkGeneration() {
    if (readGeneration() !== initialGeneration) leave();
  }
  window.addEventListener('storage', event => { if (event.key === key) leave(); });
  if (channel) channel.onmessage = event => { if (event.data === 'logout') leave(); };
  window.addEventListener('pageshow', event => {
    checkGeneration();
    // A restored document must revalidate with ngrok before showing cached data.
    if (event.persisted && !leaving) revalidate();
  });
  window.addEventListener('pagehide', () => { document.documentElement.style.visibility = 'hidden'; });
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) { checkGeneration(); revalidate(); }
  });
  function finishLogout(location) {
    if (new URLSearchParams(location.search).get('logout') === 'true') logout();
  }
  finishLogout(window.location);
  function SessionBridge() {
    const history = ReactRouter.useHistory();
    React.useEffect(() => {
      if (!subscribed) {
        subscribed = true;
        // Keep the listener when logout unmounts the app bar.
        history.listen(finishLogout);
        finishLogout(history.location);
      }
    }, [history]);
    return null;
  }
  registerAppBarAction(SessionBridge);
  revalidate();
})();
