/* Lab-only integration for Headlamp v0.46.0 behind ngrok OAuth. */
(function () {
  const { React, ReactRouter, registerAppBarAction } = window.pluginLib;
  let subscribed = false;
  function finishLogout(location) {
    if (new URLSearchParams(location.search).get('logout') === 'true') {
      window.location.replace('/ngrok/logout');
    }
  }
  finishLogout(window.location);
  function SessionBridge() {
    const history = ReactRouter.useHistory();
    React.useEffect(() => {
      if (!subscribed) {
        subscribed = true;
        // Keep this page-lifetime listener when logout unmounts the app bar.
        history.listen(finishLogout);
        finishLogout(history.location);
      }
    }, [history]);
    return null;
  }
  registerAppBarAction(SessionBridge);
})();
