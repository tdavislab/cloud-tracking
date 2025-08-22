set dataset=%1
echo Running %dataset% 
pvpython highlightCP.py %dataset%
pvpython highlightAP.py %dataset%
pvpython tracking.py %dataset%
pvpython trackingCP.py %dataset%