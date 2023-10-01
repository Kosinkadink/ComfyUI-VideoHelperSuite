import { app } from "../../../scripts/app.js";
import { api } from '../../../scripts/api.js'
import { ComfyWidgets } from "../../../scripts/widgets.js"

function videoUpload(node, inputName, inputData, app) {
    const pathWidget = node.widgets.find((w) => w.name === "video");
    const folderWidget = node.widgets.find((w) => w.name === "search_folder");

    folderWidget.content_index = 0;
    let uploadWidget;
    pathWidget.contentlists = inputData[1]
    Object.defineProperty(folderWidget, "value", {
        set : function(value) {
            this.content_index = this.options.values.indexOf(value);
            if (this.content_index == -1) {
                this.content_index = 0
            }
            pathWidget.options.values = pathWidget.contentlists[this.content_index];
            if (pathWidget.options.values.length == 0) {
                pathWidget.value = "None";
            } else {
                pathWidget.value = pathWidget.options.values[0];
            }
            this._value = value;
        },
        get : function() {
            return this._value;
        }
    });

    var default_value = "None";
    if (inputData[1][0].length > 0) {
        default_value = inputData[1][0][0];
    }
    Object.defineProperty(pathWidget, "value", {
        set : function(value) {
            if (typeof(value) == 'object') {
                //refresh event
                this.contentlists = this.options.values;
                this.options.values = this.contentlists[folderWidget.content_index];
                if (this.options.values.length > 0) {
                    value = this.options.values[0];
                } else {
                    value = "None";
                }
            }
            this._real_value = value;
        },

        get : function() {
            let value = "";
            if (this._real_value) {
                value = this._real_value;
            } else {
                return default_value;
            }

            if (value.filename) {
                let real_value = value;
                value = "";
                if (real_value.subfolder) {
                    value = real_value.subfolder + "/";
                }

                value += real_value.filename;

                if(real_value.type && real_value.type !== "input")
                    value += ` [${real_value.type}]`;
            }
            return value;
        }
    });
    async function uploadFile(file, updateNode, pasted = false) {
        try {
            // Wrap file in formdata so it includes filename
            const body = new FormData();
            body.append("image", file);
            if (pasted) body.append("subfolder", "pasted");
            const resp = await api.fetchApi("/upload/image", {
                method: "POST",
                body,
            });

            if (resp.status === 200) {
                const data = await resp.json();
                // Add the file to the dropdown list and update the widget value
                let path = data.name;
                if (data.subfolder) path = data.subfolder + "/" + path;

                if (!pathWidget.options.values.includes(path)) {
                    pathWidget.options.values.push(path);
                }

                if (updateNode) {
                    pathWidget.value = path;
                }
            } else {
                alert(resp.status + " - " + resp.statusText);
            }
        } catch (error) {
            alert(error);
        }
    }

    const fileInput = document.createElement("input");
    Object.assign(fileInput, {
        type: "file",
        accept: "video/webm,video/mp4,video/mkv,image/gif",
        style: "display: none",
        onchange: async () => {
            if (fileInput.files.length) {
                await uploadFile(fileInput.files[0], true);
            }
        },
    });
    document.body.append(fileInput);

    // Create the button widget for selecting the files
    uploadWidget = node.addWidget("button", "choose file to upload", "video", () => {
        fileInput.click();
    });
    uploadWidget.serialize = false;
    return { widget: uploadWidget };
}
ComfyWidgets.VIDEOUPLOAD = videoUpload;

app.registerExtension({
	name: "VideoHelperSuite.UploadVideo",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData?.name == "VHS_LoadVideo") {
			nodeData.input.required.upload = ["VIDEOUPLOAD", nodeData.input.required.video[0]];
            nodeData.input.required.video[1] = nodeData.input.required.video[1][0]
		}
	},
});
