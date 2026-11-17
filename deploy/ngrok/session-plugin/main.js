/* Lab-only integration for Headlamp v0.46.0 behind ngrok OAuth. */
(function () {
  const { React, ReactRouter, registerAppBarAction } = window.pluginLib;
  const key = 'swm.logout-generation';
  const readGeneration = () => { try { return localStorage.getItem(key); } catch (_) { return null; } };
  const initialGeneration = readGeneration();
  const channel = typeof BroadcastChannel === 'function' ? new BroadcastChannel('swm-session') : null;
  let subscribed = false;
  let leaving = false;
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
    if (event.persisted && !leaving) window.location.reload();
  });
  window.addEventListener('pagehide', () => { document.documentElement.style.visibility = 'hidden'; });
  document.addEventListener('visibilitychange', () => { if (!document.hidden) checkGeneration(); });
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
})();
