import { app } from '../../../scripts/app.js'

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

function addDateFormatting(nodeType, field, timestamp_widget = false) {
    chainCallback(nodeType.prototype, "onNodeCreated", function() {
        const widget = this.widgets.find((w) => w.name === "filename_prefix");
        widget.serializeValue = () => {
            return widget.value.replace(/%([^%]+)%/g, function (match, text) {
                const split = text.split(".");
                if (split[0].startsWith("date:")) {
                    return formatDate(split[0].substring(5), new Date());
                }
                return match;
            });
        };
    });
}
function addTimestampWidget(nodeType, nodeData, targetWidget) {
    const newWidgets
    for (let key in nodeData.input.required) {
        if (key == targetWidget) {
            //TODO: account for duplicate entries?
            newWidgets["timestamp_directory"] = ["BOOLEAN", {"default": true}]
        }
        newWidgets[key] = nodeData.input.required[key];
    }
    chainCallback(nodeType.prototype, "onNodeCreated", function () {
        const directoryWidget = this.widgets.find((w) => w.name === "directory_name");
        const timestampWidget = this.widgets.find((w) => w.name === "timestamp_directory");
        directoryWidget.serializeValue = () => {
            if (timestampWidget.value) {
                //ignore actual value and return timestamp
                return formatDate("yyyy-MM-ddThh:mm:ss", new Date());
            }
            return directoryWidget.value
        };
        timestampWidget._value = value;
        Object.definteProperty(timestampWidget, "value". {
            set : function(value) {
                this._value = value;
                directoryWidget.disabled = value;
            }.
            get : function() {
                return this._value;
            }
        }
    });
}

function addCustomSize(nodeType, nodeData, widgetName) {
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
                return "?x" + heightWidget;
            } else if (this.value == "Custom") {
                return widthWidget.value + "x" + heightWidget.value;
            } else {
                return this.value;
            }
        };
    });
}
function addUploadWidget(nodeType, nodeData, widgetName, type="video") {
    chainCallback(nodeType.prototype, "onNodeCreated", function() {
        const pathWidget = this.widgets.find((w) => w.name === widgetName);
        const fileInput = document.createElement("input");
        if (type == "folder") {
            Object.assign(fileInput, {
                type: "file",
                style: "display: none",
                webkitdirectory: true,
                onchange: async () => {
                    const directory = fileInput.files[0].webkitRelativePath;
                    const i = directory.lastIndexOf('/');
                    if (i <= 0) {
                        throw "No directory found";
                    }
                    const path = directory.slice(0,directory.lastIndexOf('/'))
                    if (path in pathWidget.values) {
                        alert("A folder of the same name already exists");
                        return;
                    }
                    for(const file of fileInput.files) {
                        await uploadFile(file);
                    }
                    pathWidget.values.push(path);
                    pathWidget.value = path;
                },
            });
        } else if (type == "video") {
            Object.assign(fileInput, {
                type: "file",
                accept: "video/webm,video/mp4,video/mkv,image/gif",
                style: "display: none",
                onchange: async () => {
                    if (fileInput.files.length) {
                        await uploadFile(fileInput.files[0], true);
                    }
                    const filename = fileInput.files[0].name;
                    pathWidget.values.push(filename);
                    pathWidget.value = filename;
                },
            });
        } else {
            throw "Unknown upload type"
        }
        document.body.append(fileInput);
        uploadWidget = node.addWidget("button", "choose " + type + " to upload", "image", () => {
            fileInput.click();
        });
        uploadWidget.serialize = false;
    });
}

/**
     * Defines a widget inside the node, it will be rendered on top of the node, you can control lots of properties
     *
     * @method addWidget
     * @param {String} type the widget type (could be "number","string","combo"
     * @param {String} name the text to show on the widget
     * @param {String} value the default value
     * @param {Function|String} callback function to call when it changes (optionally, it can be the name of the property to modify)
     * @param {Object} options the object that contains special properties of this widget 
     * @return {Object} the created widget object
     * LGraphNode.prototype.addWidget = function( type, name, value, callback, options )
     */

function addVideoPreview(nodeType) {
    chainCallback(nodeType,"onNodeCreated", function() {
        previewWidget = this.addWidget("preview", "videopreview");
        this.videoEl = document.createElement(type === 'video' ? 'video' : 'img');
        var x = 0;
        var width = 8;
        var y = 0;
        previewWidget.draw = (ctw, node, widget_width, y, H) => {
            //update widget position, hide if off-screen

        };
        previewWidget.computeSize = (width) => {
            return [width, width / aspect_ratio];
        }
    })
    return (src) => {};
}

app.registerExtension({
    name: "VideoHelperSuite.Core",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData?.name == "VHS_LoadImages") {
            addUploadWidget(nodeData, "directory", "folder");
        } else if (nodeData?.name == "VHS_LoadVideo") {
            addCustomSize(nodeType, nodeData, "force_size")

            addUploadWidget(nodeData, "video");
            var callback = addVideoPreview(nodeData)
        } else if (nodeData?.name == "VHS_VideoCombine") {
            addDateFormatting(nodeType, "filename_prefix");
            var callback = addVideoPreview(nodeData);

            //Hide the information passing 'gif' output
            //TODO: check how this is implemented for save image
            chainCallback(nodeType.prototype, "onNodeCreated", function() {
                this._outputs = this.outputs
                Object.defineProperty(this, "outputs", {
                    set : function(value) {
                        this._outputs = value;
                    },
                    get : function() {
                        return [];
                    }
                });
            });
        } else if (nodeData?.name == "VHS_SaveImageSequence") {
            addDateFormating(nodeType, "directory_name", timestamp_widget=true);
            addTimestampWidget(nodeType, nodeData, "directory_name")
        }
    }
});
