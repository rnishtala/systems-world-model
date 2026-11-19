/* Minimal navigation integration; all experiment permissions are server-enforced. */
(function(){const {React,registerAppBarAction}=window.pluginLib;registerAppBarAction(function(){return React.createElement('a',{href:'/experiments/',style:{color:'inherit',padding:'8px 14px',textDecoration:'none',fontSize:'14px'},title:'Train and validate a systems world model'},'Experiments ↗');});})();
