/* Namespace-only views and navigation; Kubernetes RBAC remains authoritative. */
(function(){
  const {React,registerAppBarAction}=window.pluginLib;
  const namespace='systems-world-model-lab';
  try {
    const key='cluster_settings.'+namespace;
    const settings=JSON.parse(localStorage.getItem(key)||'{}');
    if(settings.defaultNamespace!==namespace || JSON.stringify(settings.allowedNamespaces)!==JSON.stringify([namespace])) {
      localStorage.setItem(key,JSON.stringify({...settings,defaultNamespace:namespace,allowedNamespaces:[namespace],allowedNamespacesSelector:''}));
      location.reload();return;
    }
  } catch (_) { /* Browser storage may be disabled; access control still applies. */ }
  registerAppBarAction(function(){return React.createElement('a',{href:'/experiments/',style:{color:'inherit',padding:'8px 14px',textDecoration:'none',fontSize:'14px'},title:'Train and validate a systems world model'},'Experiments ↗');});
})();
