import { app } from "../../../scripts/app.js";
import { api } from '../../../scripts/api.js'
import { ComfyWidgets } from "../../../scripts/widgets.js"

function videoUpload(node, inputName, inputData, app) {
    const imageWidget = node.widgets.find((w) => w.name === "video");
    let uploadWidget;

    var default_value = imageWidget.value;
    Object.defineProperty(imageWidget, "value", {
        set : function(value) {
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

                if (!imageWidget.options.values.includes(path)) {
                    imageWidget.options.values.push(path);
                }

                if (updateNode) {
                    imageWidget.value = path;
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
	name: "VideoHelperSuit.UploadVideo",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData?.name == "VHS_LoadVideo") {
            console.log("test")
			nodeData.input.required.upload = ["VIDEOUPLOAD"];
		}
	},
});
