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

function addCustomSize(nodeType, nodeData, widgetName) {
    function chainCallback(object, property, callback) {
        if (property in object) {
            const callback_orig = object[property]
            object[property] = function () {
                const r = callback_orig.apply(this, arguments);
                callback.apply(this, arguments);
                return r
            };
        } else {
            object[property] = callback;
        }
    }

    function injectHidden(widget) {
        widget.computeSize = (target_width) => {
            if (widget.hidden) {
                return [0, -4];
            }
            return [target_width, 20];
        };
        widget._type = widget.type
        Object.defineProperty(widget, "type", {
            set : function(value) {
                widget._type = value;
            },
            get : function() {
                if (widget.hidden) {
                    return "hidden";
                }
                return widget._type;
            }
        });
    }

    //Add the extra size widgets now
    //This takes some finagling as widget order is defined by key order
    const newWidgets = {}
    for (let key in nodeData.input.required) {
        newWidgets[key] = nodeData.input.required[key]
        if (key == widgetName) {
            newWidgets[key][0] = newWidgets[key][0].concat(["Custom Width", "Custom Height", "Custom"])
            newWidgets["custom_width"] = ["INT", {"default": 512, "min": 8, "step": 8}]
            newWidgets["custom_height"] = ["INT", {"default": 512, "min": 8, "step": 8}]
        }
    }
    nodeData.input.required = newWidgets;

    //Add a callback which sets up the actual logic once the node is created
    chainCallback(nodeType.prototype, "onNodeCreated", function() {
        const node = this;
        const sizeOptionWidget = node.widgets.find((w) => w.name === widgetName);
        const widthWidget = node.widgets.find((w) => w.name === "custom_width");
        const heightWidget = node.widgets.find((w) => w.name === "custom_height");
        injectHidden(widthWidget);
        widthWidget.serialize = false;
        injectHidden(heightWidget);
        heightWidget.serialize = false;
        sizeOptionWidget._value = sizeOptionWidget.value;
        Object.defineProperty(sizeOptionWidget, "value", {
            set : function(value) {
                //TODO: Only modify hidden/reset size when a change occurs
                if (value == "Custom Width") {
                    widthWidget.hidden = false;
                    heightWidget.hidden = true;
                } else if (value == "Custom Height") {
                    widthWidget.hidden = true;
                    heightWidget.hidden = false;
                } else if (value == "Custom") {
                    widthWidget.hidden = false;
                    heightWidget.hidden = false;
                } else{
                    widthWidget.hidden = true;
                    heightWidget.hidden = true;
                }
                node.setSize([node.size[0], node.computeSize([node.size[0], node.size[1]])[1]])
                this._value = value;
            },
            get : function() {
                return this._value;
            }
        });
        //Ensure proper visibility/size state for initial value
        sizeOptionWidget.value = sizeOptionWidget._value;

        sizeOptionWidget.serializeValue = function() {
            if (this.value == "Custom Width") {
                return widthWidget.value + "x?";
            } else if (this.value == "Custom Height") {
                return "?x" + heightWidget.value;
            } else if (this.value == "Custom") {
                return widthWidget.value + "x" + heightWidget.value;
            } else {
                return this.value;
            }
        };
    });
}

app.registerExtension({
	name: "VideoHelperSuite.UploadVideo",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		switch (nodeData?.name) {
			case "VHS_LoadVideo":
                addCustomSize(nodeType, nodeData, "force_size");
				// Fall into next case
			case 'VHS_UploadVideo': {
				nodeData.input.required.upload = ["VIDEOUPLOAD"];
			}
		}
	}
});
