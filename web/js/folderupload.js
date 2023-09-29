import { app } from "../../../scripts/app.js";
import { api } from '../../../scripts/api.js'
import { ComfyWidgets } from "../../../scripts/widgets.js"

function folderUpload(node, inputName, inputData, app) {
    const directoryWidget = node.widgets.find((w) => w.name === "directory");
    let uploadWidget;

    var default_value = directoryWidget.value;
    async function uploadFile(file) {
        try {
            // Wrap file in formdata so it includes filename
            const body = new FormData();
            const i = file.webkitRelativePath.lastIndexOf('/');
            const subfolder = file.webkitRelativePath.slice(0,i+1)
            const new_file = new File([file], file.name, {
                type: file.type,
                lastModified: file.lastModified,
            });
            body.append("image", new_file);
            if (i > 0) {
                body.append("subfolder", subfolder);
            }
            const resp = await api.fetchApi("/upload/image", {
                method: "POST",
                body,
            });

            if (resp.status === 200) {
                const data = await resp.json();
                // Add the new folder to the dropdown list and update the widget value
                if (!directoryWidget.options.values.includes(data.subfolder)) {
                    directoryWidget.options.values.push(subfolder);
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
        style: "display: none",
        webkitdirectory: true,
        onchange: async () => {
            for(const file of fileInput.files) {
                await uploadFile(file);
            }
            const directory = fileInput.files[0].webkitRelativePath
            const i = directory.lastIndexOf('/');
            if (i > 0) {
                directoryWidget.value = directory.slice(0,directory.lastIndexOf('/'))
            }
            app.file = fileInput.files[0]
        },
    });
    document.body.append(fileInput);

    // Create the button widget for selecting the files
    uploadWidget = node.addWidget("button", "choose folder to upload", "directory", () => {
        fileInput.click();
    });
    uploadWidget.serialize = false;
    return { widget: uploadWidget };
}
ComfyWidgets.FOLDERUPLOAD = folderUpload;

app.registerExtension({
	name: "VideoHelperSuite.UploadFolder",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData?.name == "VHS_LoadImages") {
			nodeData.input.required.upload = ["FOLDERUPLOAD"];
		}
	},
});
