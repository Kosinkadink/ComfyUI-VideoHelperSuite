import server
import folder_paths
import os
import subprocess

web = server.web

@server.PromptServer.instance.routes.get("/getpath")
async def get_path(request):
    query = request.rel_url.query
    if "path" not in query:
        return web.Response(status=404)
    path = os.path.abspath(query["path"])

    #For now, only continue if subpath of comfui
    basedir = os.path.abspath('.')
    common_path = os.path.commonpath([basedir, path])
    if common_path != basedir or not os.path.exists(path):
        return web.Response(status=404)

    #Use get so None is default instead of keyerror
    valid_extensions = query.get("extensions")
    valid_items = []
    for item in os.scandir(path):
        #TODO: change type to designate if dir or file
        if item.is_dir():
            valid_items.append(item.name + "/")
            continue
        if valid_extensions is None or item.name.split(".")[-1] in valid_extensions:
            valid_items.append(item.name)

    return web.json_response(valid_items)
