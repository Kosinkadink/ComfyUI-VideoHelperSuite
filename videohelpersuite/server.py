import server
import folder_paths
import os
import subprocess

web = server.web

@server.PromptServer.instance.routes.get("/viewvideo")
async def view_video(request):
    query = request.rel_url.query
    if "filename" not in query:
        return web.Response(status=404)
    filename = query["filename"]
    filename, output_dir = folder_paths.annotated_filepath(filename)

    filename,output_dir = folder_paths.annotated_filepath(filename)

    # validation for security: prevent accessing arbitrary path
    if filename[0] == '/' or '..' in filename:
        return web.Response(status=400)

    if output_dir is None:
        type = request.rel_url.query.get("type", "output")
        output_dir = folder_paths.get_directory_by_type(type)

    if output_dir is None:
        return web.Response(status=400)

    if "subfolder" in request.rel_url.query:
        full_output_dir = os.path.join(output_dir, request.rel_url.query["subfolder"])
        if os.path.commonpath((os.path.abspath(full_output_dir), output_dir)) != output_dir:
            return web.Response(status=403)
        output_dir = full_output_dir

    filename = os.path.basename(filename)
    file = os.path.join(output_dir, filename)

    if not os.path.isfile(file):
        return web.Response(status=404)

    with subprocess.Popen(["ffmpeg", "-v", "error", "-i", file, "-t", "10", "-c:v", "copy", "-f", "webm", "-"], stdout=subprocess.PIPE) as proc:
        resp = web.StreamResponse()
        resp.content_type = 'video/webm'
        resp.headers["Content-Disposition"] = f"filename=\"{filename}\""
        await resp.prepare(request)
        while True:
            bytes_read = proc.stdout.read()
            if bytes_read is None:
                #TODO: check for timeout here
                time.sleep(.1)
                continue
            if len(bytes_read) == 0:
                break
            await resp.write(bytes_read)
        return resp
