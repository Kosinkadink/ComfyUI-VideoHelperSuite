import { app } from '../../../scripts/app.js'
import { api } from '../../../scripts/api.js'

function chainCallback(object, property, callback) {
    if (object == undefined) {
        //This should not happen.
        debugger;
        return;
    }
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

function useKVState(nodeType) {
    chainCallback(nodeType.prototype, "onNodeCreated", function () {
        chainCallback(this, "onConfigure", function(info) {
            if (typeof(info.widgets_values) == 'object' && info.widgets_values.length == undefined) {
                for (let key in info.widgets_values) {
                    let w = this.widgets.find((w) => w.name == key);
                    if (w == undefined) {
                        //widget with name does not exist. removal/old version?
                        continue
                    }
                    w.value = info.widgets_values[key];
                }
            } else {
                //Saved data was not a map made by this method
                //This probably means it's an array and was already restored.
                if (info?.widgets_values?.length != this.widgets.length) {
                    //Widget could not have restored properly

                    //First try a number of known migrations.
                    //This code hits only specific common cases and is deprecated
                    if(this.type == "VHS_LoadVideo" && info?.widgets_values?.length == 7) {
                        //custom_width, custom_height, preview added. Shift and load
                        this.widgets[3].value = 512;
                        this.widgets[4].value = 512;
                        this.widgets[5].value = info.widgets_values[3];
                        this.widgets[6].value = info.widgets_values[4];
                        this.widgets[7].value = info.widgets_values[5];
                        return
                    } else if (this.type == "VHS_VideoCombine" && info?.widgets_values?.length == 7) {
                        //the sync button was removed, but actual loading was not impacted
                        return
                    } else if (this.type == "VHS_VideoCombine" && info?.widgets_values?.length == 8) {
                        //single active preview
                        if (info.widgets_values[7].startsWith("/view?")) {
                            return
                        }
                    }

                    //Note if multiple node loads fail, only the latest error dialog displays
                    app.ui.dialog.show("Failed to restore node: " + this.title + "\nPlease remove and re-add it.")
                    this.bgcolor = "#C00"
                }
            }
        });
        chainCallback(this, "onSerialize", function(info) {
            info.widgets_values = {};
            for (let w of this.widgets) {
                info.widgets_values[w.name] = w.value;
            }
        });
    })
}

function fitHeight(node) {
    node.setSize([node.size[0], node.computeSize([node.size[0], node.size[1]])[1]])
    node.graph.setDirtyCanvas(true);
}

async function uploadFile(file) {
    //TODO: Add uploaded file to cache with Cache.put()?
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
            return resp.status
        } else {
            alert(resp.status + " - " + resp.statusText);
        }
    } catch (error) {
        alert(error);
    }
}

function formatDate(text, date) {
    const parts = {
        d: (d) => d.getDate(),
        M: (d) => d.getMonth() + 1,
        h: (d) => d.getHours(),
        m: (d) => d.getMinutes(),
        s: (d) => d.getSeconds(),
    };
    const format =
        Object.keys(parts)
        .map((k) => k + k + "?")
        .join("|") + "|yyy?y?";
    return text.replace(new RegExp(format, "g"), function (text) {
        if (text === "yy") return (date.getFullYear() + "").substring(2);
        if (text === "yyyy") return date.getFullYear();
        if (text[0] in parts) {
            const p = parts[text[0]](date);
            return (p + "").padStart(text.length, "0");
        }
        return text;
    });
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
    const newWidgets = {};
    for (let key in nodeData.input.required) {
        if (key == targetWidget) {
            //TODO: account for duplicate entries?
            newWidgets["timestamp_directory"] = ["BOOLEAN", {"default": true}]
        }
        newWidgets[key] = nodeData.input.required[key];
    }
    nodeDta.input.required = newWidgets;
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
        Object.definteProperty(timestampWidget, "value", {
            set : function(value) {
                this._value = value;
                directoryWidget.disabled = value;
            },
            get : function() {
                return this._value;
            }
        });
    });
}

function addCustomSize(nodeType, nodeData, widgetName) {
    //Add the extra size widgets now
    //This takes some finagling as widget order is defined by key order
    const newWidgets = {};
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
        widthWidget.options.serialize = false;
        injectHidden(heightWidget);
        heightWidget.options.serialize = false;
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
        chainCallback(this, "onRemoved", () => {
            fileInput?.remove();
        });
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
                    if (pathWidget.options.values.includes(path)) {
                        alert("A folder of the same name already exists");
                        return;
                    }
                    let successes = 0;
                    for(const file of fileInput.files) {
                        if (await uploadFile(file) == 200) {
                            successes++;
                        } else {
                            //Upload failed, but some prior uploads may have succeeded
                            //Stop future uploads to prevent cascading failures
                            //and only add to list if an upload has succeeded
                            if (successes > 0) {
                                break
                            } else {
                                return;
                            }
                        }
                    }
                    pathWidget.options.values.push(path);
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
                        if (await uploadFile(fileInput.files[0]) != 200) {
                            //upload failed and file can not be added to options
                            return;
                        }
                    }
                    const filename = fileInput.files[0].name;
                    pathWidget.options.values.push(filename);
                    pathWidget.value = filename;
                },
            });
        } else {
            throw "Unknown upload type"
        }
        document.body.append(fileInput);
        let uploadWidget = this.addWidget("button", "choose " + type + " to upload", "image", () => {
            fileInput.click();
        });
        uploadWidget.options.serialize = false;
    });
}

function addVideoPreview(nodeType) {
    chainCallback(nodeType.prototype, "onNodeCreated", function() {
        //preview is a made up widget type to enable user defined functions
        //videopreview is widget name
        //The previous implementation used type to distinguish between a video and gif,
        //but the type is not serialized and would not survive a reload
        var previewWidget = { name : "videopreview", type : "preview", value : "",
            draw : function(ctx, node, widgetWidth, widgetY, height) {
                //update widget position, hide if off-screen
                const transform = ctx.getTransform();
                const x = transform.e;
                const y = transform.f;
                this._currentwidth = (widgetWidth-30);
                const scale = transform.a;//scale x and scale y always equal
                Object.assign(this.parentEl.style, {
                    left: (x+15*scale) + "px",
                    top: (y + widgetY*scale) + "px",
                    width: ((widgetWidth-30)*scale) + "px",
                    zIndex: 2 + (node.is_selected ? 1 : 0),
                    position: "absolute",
                });
                this._boundingCount = 0;
            },
            computeSize : function(width) {
                if (this.aspectRatio && !this.parentEl.hidden) {
                    return [width, this._currentwidth / this.aspectRatio];
                }
                return [width, -4];//no loaded src, widget should not display
            }
        };
        //onRemoved isn't a litegraph supported function on widgets
        //Given that onremoved widget and node callbacks are sparse, this
        //saves the required iteration.
        chainCallback(this, "onRemoved", () => {
            previewWidget?.parentEl?.remove();
        });
        this.addCustomWidget(previewWidget);
        previewWidget.parentEl = document.createElement("div");
        previewWidget.parentEl.className = "vhs_preview";
        previewWidget.parentEl.style['pointer-events'] = "none"

        previewWidget.videoEl = document.createElement("video");
        previewWidget.videoEl.controls = false;
        previewWidget.videoEl.autoplay = true;
        previewWidget.videoEl.loop = true;
        previewWidget.videoEl.muted = true;
        previewWidget.videoEl.style['width'] = "100%"
        previewWidget.videoEl.addEventListener("loadedmetadata", () => {
            previewWidget.aspectRatio = previewWidget.videoEl.videoWidth / previewWidget.videoEl.videoHeight;
            fitHeight(this);
        });
        previewWidget.videoEl.addEventListener("error", () => {
            //TODO: consider a way to properly notify the user why a preview isn't shown.
            previewWidget.parentEl.hidden = true;
            fitHeight(this);
        });

        previewWidget.imgEl = document.createElement("img");
        previewWidget.imgEl.style['width'] = "100%"
        previewWidget.imgEl.hidden = true;
        previewWidget.imgEl.onload = () => {
            previewWidget.aspectRatio = previewWidget.imgEl.naturalWidth / previewWidget.imgEl.naturalHeight;
            fitHeight(this);
        };

        this.setPreviewsrc = function(params) {
            previewWidget.parentEl.hidden = false;
            //example url for testing
            //http://127.0.0.1:8188/view?filename=leader.webm&subfolder=&type=input&format=video%2Fwebm
            if (params?.format?.split('/')[0] == 'video') {
                previewWidget.videoEl.src = api.apiURL('/view?' + new URLSearchParams(params));
                previewWidget.videoEl.hidden = false;
                previewWidget.imgEl.hidden = true;
            } else {
                //Is animated image
                previewWidget.imgEl.src = api.apiURL('/view?' + new URLSearchParams(params));
                previewWidget.videoEl.hidden = true;
                previewWidget.imgEl.hidden = false;
            }

        }
        //Hide video element if offscreen
        //The multiline input implementation moves offscreen every frame
        //and doesn't apply until a node with an actual inputEl is loaded
        this._boundingCount = 0;
        this.onBounding = function() {
            if (this._boundingCount++>5) {
                previewWidget.parentEl.style.left = "-8000px";
            }
        }
        //this.setPreviewsrc({filename : "leader.webm", type : "input", format: "video/webm"})
        previewWidget.parentEl.appendChild(previewWidget.videoEl)
        previewWidget.parentEl.appendChild(previewWidget.imgEl)
        document.body.appendChild(previewWidget.parentEl);
    });
}
function addPreviewOptions(nodeType) {
    chainCallback(nodeType.prototype, "getExtraMenuOptions", function(_, options) {
        // The intended way of appending options is returning a list of extra options,
        // but this isn't used in widgetInputs.js and would require
        // less generalization of chainCallback
        let optNew = []
        const previewWidget = this.widgets.find((w) => w.name === "videopreview");

        let url = null
        if (previewWidget.videoEl?.hidden == false && previewWidget.videoEl.src) {
            url = previewWidget.videoEl.src;
        } else if (previewWidget.imgEl?.hidden == false && previewWidget.imgEl.src) {
            url = previewWidget.imgEl.src;
        }
        if (url) {
            url = new URL(url);
            //placeholder from Save Image, will matter once preview functionality is implemented
            //url.searchParams.delete('preview')
            optNew.push(
                {
                    content: "Open preview",
                    callback: () => {
                        window.open(url, "_blank")
                    },
                },
                {
                    content: "Save preview",
                    callback: () => {
                        const a = document.createElement("a");
                        a.href = url;
                        a.setAttribute("download", new URLSearchParams(url.search).get("filename"));
                        document.body.append(a);
                        a.click();
                        requestAnimationFrame(() => a.remove());
                    },
                }
            );
        }
        const PauseDesc = (previewWidget.paused ? "Resume" : "Pause") + " preview";
        if(previewWidget.videoEl.hidden == false) {
            optNew.push({content: PauseDesc, callback: () => {
                //animated images can't be paused and are more likely to cause performance issues.
                //changing src to a single keyframe is possible,
                //For now, the option is disabled if an animated image is being displayed
                if(previewWidget.paused) {
                    previewWidget.paused = false;
                    previewWidget.videoEl?.play();
                } else {
                    previewWidget.paused = true;
                    previewWidget.videoEl?.pause();
                }
            }});
        }
        //TODO: Consider hiding elements if video no preview is available yet.
        //It would reduce confusion at the cost of functionality
        //(if a video preview lags the computer, the user should be able to hide in advance)
        const visDesc = (previewWidget.parentEl.hidden ? "Show" : "Hide") + " preview";
        optNew.push({content: visDesc, callback: () => {
            previewWidget.parentEl.hidden = !previewWidget.parentEl.hidden;
            fitHeight(this);

        }});
        optNew.push({content: "Sync preview", callback: () => {
            //TODO: address case where videos have varying length
            //Consider a system of sync groups which are opt-in?
            for (let p of document.getElementsByClassName("vhs_preview")) {
                for (let child of p.children) {
                    if (child.tagName == "VIDEO") {
                        child.currentTime=0;
                    } else if (child.tagName == "IMG") {
                        child.src = child.src;
                    }
                }
            }
        }});
        if(options.length > 0 && options[0] != null && optNew.length > 0) {
            optNew.push(null);
        }
        options.unshift(...optNew);
    });
}

app.registerExtension({
    name: "VideoHelperSuite.Core",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if(nodeData?.name?.startsWith("VHS_")) {
            useKVState(nodeType);
        }
        if (nodeData?.name == "VHS_LoadImages") {
            addUploadWidget(nodeType, nodeData, "directory", "folder");
        } else if (nodeData?.name == "VHS_LoadVideo") {
            addCustomSize(nodeType, nodeData, "force_size")

            addUploadWidget(nodeType, nodeData, "video");
            addVideoPreview(nodeType);
            addPreviewOptions(nodeType);
            chainCallback(nodeType.prototype, "onNodeCreated", function() {
                const pathWidget = this.widgets.find((w) => w.name === "video");
                pathWidget._value = pathWidget.value;
                Object.defineProperty(pathWidget, "value", {
                    set : (value) => {
                        pathWidget._value = value;
                        //let parts = value.split("//");
                        let parts = ["input", value];
                        let extension_index = parts[1].lastIndexOf(".");
                        let extension = parts[1].slice(extension_index+1);
                        let format = "video"
                        if (["gif", "webp", "avif"].includes(extension)) {
                            format = "image"
                        }
                        this.setPreviewsrc({filename : parts[1], type : parts[0], format: format});
                    },
                    get : () => {
                        return pathWidget._value;
                    }
                });
                //Set value to ensure preview displays on initial add.
                pathWidget.value = pathWidget._value;
            });
        } else if (nodeData?.name == "VHS_VideoCombine") {
            addDateFormatting(nodeType, "filename_prefix");
            chainCallback(nodeType.prototype, "onExecuted", function(message) {
                if (message?.gifs) {
                    this.setPreviewsrc(message.gifs[0]);
                }
            });
            addVideoPreview(nodeType);
            addPreviewOptions(nodeType);

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
            //Disabled for safety as VHS_SaveImageSequence is not currently merged
            //addDateFormating(nodeType, "directory_name", timestamp_widget=true);
            //addTimestampWidget(nodeType, nodeData, "directory_name")
        }
    }
});
