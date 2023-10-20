import { app } from "../../../scripts/app.js";
import { api } from '../../../scripts/api.js'
import { ComfyWidgets } from "../../../scripts/widgets.js"
import { acceptableFileTypes } from "./gif_preview.js"

function videoUpload(node, inputName, inputData, app) {
	const videoWidget = node.widgets.find((w) => w.name === "video");
	const typeWidget = node.widgets.find((w) => w.name === "upload_to_directory");
	let uploadWidget;

	console.log(videoWidget);

	// Clear widget value if temp since it didn't survive the relaunch
	if (videoWidget.value != undefined && videoWidget.value.startsWith("temp/")) {
		videoWidget.value = "";
	}

	var default_value = videoWidget.value;
	Object.defineProperty(videoWidget, "value", {
		set: function(value) {
			this._real_value = value;
		},

		get: function() {
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

				if (real_value.type && real_value.type !== "input")
					value += ` [${real_value.type}]`;
			}
			return value;
		}
	});
	async function uploadFile(file, node, updateNode, pasted = false) {
		try {
			// Wrap file in formdata so it includes filename
			const body = new FormData();
			body.append("image", file);
			body.append("type", typeWidget.value);
			body.append("subfolder", "VHS_upload");
			const resp = await api.fetchApi("/upload/image", {
				method: "POST",
				body,
			});

			if (resp.status === 200) {
				const data = await resp.json();
				// Add the file to the dropdown list and update the widget value
				let path = data.name;
				if (data.subfolder) path = data.subfolder + "/" + path;
				if (data.type) path = data.type + "/" + path;

				if (!videoWidget.options.values.includes(path)) {
					videoWidget.options.values.push(path);
				}

				if (updateNode) {
					videoWidget.value = path;
					videoWidget.callback(path); // Update video container
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
		accept: acceptableFileTypes.join(","),
		style: "display: none",
		onchange: async () => {
			if (fileInput.files.length) {
				await uploadFile(fileInput.files[0], node, true);
			}
		},
	});
	document.body.append(fileInput);

	// Create the button widget for selecting the files
	uploadWidget = node.addWidget("button", "choose file or drag and drop to upload", "video", () => {
		fileInput.click();
	});
	uploadWidget.serialize = false;

	// Add handler to check if an image is being dragged over our node
	node.onDragOver = function(e) {
		if (e.dataTransfer && e.dataTransfer.items) {
			// Check if any of the dragged files are images or videos
			for (const file of e.dataTransfer.items) {
				if (acceptableFileTypes.includes(file.type)) {
					return true;
				}
			}
		}

		return false;
	};

	// On drop upload files
	node.onDragDrop = function(e) {
		if (e.dataTransfer && e.dataTransfer.files) {
			for (const file of e.dataTransfer.files) {
				if (acceptableFileTypes.includes(file.type)) {
					uploadFile(file, node, true); // Just upload the very first matching object in the payload
					return true;
				}
			}
		}

		return false;
	};

	return { widget: uploadWidget };
}

ComfyWidgets.VIDEOUPLOAD = videoUpload;

app.registerExtension({
	name: "VideoHelperSuite.UploadVideo",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		switch (nodeData?.name) {
			case "VHS_LoadVideo":
				// Fall into next case
			case 'VHS_UploadVideo': {
				nodeData.input.required.upload = ["VIDEOUPLOAD"];
			}
		}
	}
});