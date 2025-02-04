import { app } from '../../../scripts/app.js'
import { api } from '../../../scripts/api.js'
import { setWidgetConfig } from '../../../extensions/core/widgetInputs.js'
import { applyTextReplacements } from "../../../scripts/utils.js";

function chainCallback(object, property, callback) {
    if (object == undefined) {
        //This should not happen.
        console.error("Tried to add callback to non-existant object")
        return;
    }
    if (property in object && object[property]) {
        const callback_orig = object[property]
        object[property] = function () {
            const r = callback_orig.apply(this, arguments);
            return callback.apply(this, arguments) ?? r
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

const convDict = {
    VHS_LoadImages : ["directory", null, "image_load_cap", "skip_first_images", "select_every_nth"],
    VHS_LoadImagesPath : ["directory", "image_load_cap", "skip_first_images", "select_every_nth"],
    VHS_VideoCombine : ["frame_rate", "loop_count", "filename_prefix", "format", "pingpong", "save_image"],
    VHS_LoadVideo : ["video", "force_rate", "force_size", "frame_load_cap", "skip_first_frames", "select_every_nth"],
    VHS_LoadVideoPath : ["video", "force_rate", "force_size", "frame_load_cap", "skip_first_frames", "select_every_nth"],
};
const renameDict  = {VHS_VideoCombine : {save_output : "save_image"}}
function useKVState(nodeType) {
    chainCallback(nodeType.prototype, "onNodeCreated", function () {
        chainCallback(this, "onConfigure", function(info) {
            if (!this.widgets) {
                //Node has no widgets, there is nothing to restore
                return
            }
            if (typeof(info.widgets_values) != "object") {
                //widgets_values is in some unknown inactionable format
                return
            }
            let widgetDict = info.widgets_values
            if (info.widgets_values.length) {
                //widgets_values is in the old list format
                if (this.type in convDict) {
                    //widget does not have a conversion format provided
                    let convList = convDict[this.type];
                    if(info.widgets_values.length >= convList.length) {
                        //has all required fields
                        widgetDict = {}
                        for (let i = 0; i < convList.length; i++) {
                            if(!convList[i]) {
                                //Element should not be processed (upload button on load image sequence)
                                continue
                            }
                            widgetDict[convList[i]] = info.widgets_values[i];
                        }
                    } else {
                        //widgets_values is missing elements marked as required
                        //let it fall through to failure state
                    }
                }
            }
            if ('force_size' in widgetDict) {
                //force size has been phased out, Migrate state
                if (widgetDict.force_size.includes?.('x')) {
                    let sizes = widgetDict.force_size.split('x')
                    if (sizes[0] != '?') {
                        widgetDict.custom_width = parseInt(sizes[0])
                    } else {
                        widgetDict.custom_width = 0
                    }
                    if (sizes[1] != '?') {
                        widgetDict.custom_height = parseInt(sizes[1])
                    } else {
                        widgetDict.custom_height = 0
                    }
                } else {
                    if (['Disabled', 'Custom Height'].includes(widgetDict.force_size)) {
                        widgetDict.custom_width = 0
                    }
                    if (['Disabled', 'Custom Width'].includes(widgetDict.force_size)) {
                        widgetDict.custom_height = 0
                    }
                }
            }
            if (widgetDict.videopreview?.params?.force_size) {
                delete widgetDict.videopreview.params.force_size
            }
            let inputs = {}
            for (let i of this.inputs) {
                inputs[i.name] = i
            }
            if (widgetDict.length == undefined) {
                for (let w of this.widgets) {
                    if (w.type =="button") {
                        continue
                    }
                    if (w.name in widgetDict) {
                        w.value = widgetDict[w.name];
                        w.callback?.(w.value)
                    } else {
                        //Check for a legacy name that needs migrating
                        if (this.type in renameDict && w.name in renameDict[this.type]) {
                            if (renameDict[this.type][w.name] in widgetDict) {
                                w.value = widgetDict[renameDict[this.type][w.name]]
                                w.callback?.(w.value)
                                continue
                            }
                        }
                        //attempt to restore default value
                        let inputs = LiteGraph.getNodeType(this.type).nodeData.input;
                        let initialValue = null;
                        if (inputs?.required?.hasOwnProperty(w.name)) {
                            if (inputs.required[w.name][1]?.hasOwnProperty("default")) {
                                initialValue = inputs.required[w.name][1].default;
                            } else if (inputs.required[w.name][0].length) {
                                initialValue = inputs.required[w.name][0][0];
                            }
                        } else if (inputs?.optional?.hasOwnProperty(w.name)) {
                            if (inputs.optional[w.name][1]?.hasOwnProperty("default")) {
                                initialValue = inputs.optional[w.name][1].default;
                            } else if (inputs.optional[w.name][0].length) {
                                initialValue = inputs.optional[w.name][0][0];
                            }
                        }
                        if (initialValue) {
                            w.value = initialValue;
                            w.callback?.(w.value)
                        }
                    }
                    if (w.name in inputs && w.config) {
                        setWidgetConfig(inputs[w.name], w.config, w)
                    }
                }
            } else {
                //Saved data was not a map made by this method
                //and a conversion dict for it does not exist
                //It's likely an array and that has been blindly applied
                if (info?.widgets_values?.length != this.widgets.length) {
                    //Widget could not have restored properly
                    //Note if multiple node loads fail, only the latest error dialog displays
                    app.ui.dialog.show("Failed to restore node: " + this.title + "\nPlease remove and re-add it.")
                    this.bgcolor = "#C00"
                }
            }
        });
        chainCallback(this, "onSerialize", function(info) {
            info.widgets_values = {};
            if (!this.widgets) {
                //object has no widgets, there is nothing to store
                return;
            }
            for (let w of this.widgets) {
                info.widgets_values[w.name] = w.value;
            }
        });
    })
}
var helpDOM;
if (!app.helpDOM) {
    helpDOM = document.createElement("div");
    app.VHSHelp = helpDOM
}
function initHelpDOM() {
    let parentDOM = document.createElement("div");
    parentDOM.className = "VHS_floatinghelp"
    document.body.appendChild(parentDOM)
    parentDOM.appendChild(helpDOM)
    helpDOM.className = "litegraph";
    let scrollbarStyle = document.createElement('style');
    scrollbarStyle.innerHTML = `
            .VHS_floatinghelp {
                scrollbar-width: 6px;
                scrollbar-color: #0003  #0000;
                &::-webkit-scrollbar {
                    background: transparent;
                    width: 6px;
                }
                &::-webkit-scrollbar-thumb {
                    background: #0005;
                    border-radius: 20px
                }
                &::-webkit-scrollbar-button {
                    display: none;
                }
            }
            .VHS_loopedvideo::-webkit-media-controls-mute-button {
                display:none;
            }
            .VHS_loopedvideo::-webkit-media-controls-fullscreen-button {
                display:none;
            }
    `
    scrollbarStyle.id = 'scroll-properties'
    parentDOM.appendChild(scrollbarStyle)
    chainCallback(app.canvas, "onDrawForeground", function (ctx, visible_rect){
        let n = helpDOM.node
        if (!n || !n?.graph) {
            parentDOM.style['left'] = '-5000px'
            return
        }
        //draw : function(ctx, node, widgetWidth, widgetY, height) {
        //update widget position, even if off screen
        const transform = ctx.getTransform();
        const scale = app.canvas.ds.scale;//gets the litegraph zoom
        //calculate coordinates with account for browser zoom
        const bcr = app.canvas.canvas.getBoundingClientRect()
        const x = transform.e*scale/transform.a + bcr.x;
        const y = transform.f*scale/transform.a + bcr.y;
        //TODO: text reflows at low zoom. investigate alternatives
        Object.assign(parentDOM.style, {
            left: (x+(n.pos[0] + n.size[0]+15)*scale) + "px",
            top: (y+(n.pos[1]-LiteGraph.NODE_TITLE_HEIGHT)*scale) + "px",
            width: "400px",
            minHeight: "100px",
            maxHeight: "600px",
            overflowY: 'scroll',
            transformOrigin: '0 0',
            transform: 'scale(' + scale + ',' + scale +')',
            fontSize: '18px',
            backgroundColor: LiteGraph.NODE_DEFAULT_BGCOLOR,
            boxShadow: '0 0 10px black',
            borderRadius: '4px',
            padding: '3px',
            zIndex: 3,
            position: "absolute",
            display: 'inline',
        });
    });
    function setCollapse(el, doCollapse) {
        if (doCollapse) {
            el.children[0].children[0].innerHTML = '+'
            Object.assign(el.children[1].style, {
                color: '#CCC',
                overflowX: 'hidden',
                width: '0px',
                minWidth: 'calc(100% - 20px)',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
            })
            for (let child of el.children[1].children) {
                if (child.style.display != 'none'){
                    child.origDisplay = child.style.display
                }
                child.style.display = 'none'
            }
        } else {
            el.children[0].children[0].innerHTML = '-'
            Object.assign(el.children[1].style, {
                color: '',
                overflowX: '',
                width: '100%',
                minWidth: '',
                textOverflow: '',
                whiteSpace: '',
            })
            for (let child of el.children[1].children) {
                child.style.display = child.origDisplay
            }
        }
    }
    helpDOM.collapseOnClick = function() {
        let doCollapse = this.children[0].innerHTML == '-'
        setCollapse(this.parentElement, doCollapse)
    }
    helpDOM.selectHelp = function(name, value) {
        //attempt to navigate to name in help
        function collapseUnlessMatch(items,t) {
            var match = items.querySelector('[vhs_title="' + t + '"]')
            if (!match) {
                for (let i of items.children) {
                    if (i.innerHTML.slice(0,t.length+5).includes(t)) {
                        match = i
                        break
                    }
                }
            }
            if (!match) {
                return null
            }
            //For longer documentation items with fewer collapsable elements,
            //scroll to make sure the entirety of the selected item is visible
            //This has the unfortunate side effect of trying to scroll the main
            //window if the documentation windows is forcibly offscreen,
            //but it's easy to simply scroll the main window back and seems to
            //have no visual side effects
            match.scrollIntoView(false)
            window.scrollTo(0,0)
            for (let i of items.querySelectorAll('.VHS_collapse')) {
                if (i.contains(match)) {
                    setCollapse(i, false)
                } else {
                    setCollapse(i, true)
                }
            }
            return match
        }
        let target = collapseUnlessMatch(helpDOM, name)
        if (target && value) {
            collapseUnlessMatch(target, value)
        }
    }
    let titleContext = document.createElement("canvas").getContext("2d")
    titleContext.font = app.canvas.title_text_font;
    helpDOM.calculateTitleLength = function(text) {
        return titleContext.measureText(text).width
    }
    helpDOM.addHelp = function(node, nodeType, description) {
        if (!description) {
            return
        }
        //Pad computed size for the clickable question mark
        let originalComputeSize = node.computeSize
        node.computeSize = function() {
            let size = originalComputeSize.apply(this, arguments)
            if (!this.title) {
                return size
            }
            let title_width = helpDOM.calculateTitleLength(this.title)
            size[0] = Math.max(size[0], title_width + LiteGraph.NODE_TITLE_HEIGHT*2)
            return size
        }

        node.description = description
        chainCallback(node, "onDrawForeground", function (ctx) {
            if (this?.flags?.collapsed) {
                return
            }
            //draw question mark
            ctx.save()
            ctx.font = 'bold 20px Arial'
            ctx.fillText("?", this.size[0]-17, -8)
            ctx.restore()
        })
        chainCallback(node, "onMouseDown", function (e, pos, canvas) {
            if (this?.flags?.collapsed) {
                return
            }
            //On click would be preferred, but this'll be good enough
            if (pos[1] < 0 && pos[0] + LiteGraph.NODE_TITLE_HEIGHT > this.size[0]) {
                //corner question mark clicked
                if (helpDOM.node == this) {
                    helpDOM.node = undefined
                } else {
                    helpDOM.node = this;
                    helpDOM.innerHTML = this.description || "no help provided "
                    for (let e of helpDOM.querySelectorAll('.VHS_collapse')) {
                        e.children[0].onclick = helpDOM.collapseOnClick
                        e.children[0].style.cursor = 'pointer'
                    }
                    for (let e of helpDOM.querySelectorAll('.VHS_precollapse')) {
                        setCollapse(e, true)
                    }
                    for (let e of helpDOM.querySelectorAll('.VHS_loopedvideo')) {
                        e?.play()
                    }
                    helpDOM.parentElement.scrollTo(0,0)
                }
                return true
            }
        })
        let timeout = null
        chainCallback(node, "onMouseMove", function (e, pos, canvas) {
            if (timeout) {
                clearTimeout(timeout)
                timeout = null
            }
            if (helpDOM.node != this) {
                return
            }
            timeout = setTimeout(() => {
                let n = this
                if (pos[0] > 0 && pos[0] < n.size[0]
                    && pos[1] > 0 && pos[1] < n.size[1]) {
                    //TODO: provide help specific to element clicked
                    let inputRows = Math.max(n.inputs?.length || 0, n.outputs?.length || 0)
                    if (pos[1] < LiteGraph.NODE_SLOT_HEIGHT * inputRows) {
                        let row = Math.floor((pos[1] - 7) / LiteGraph.NODE_SLOT_HEIGHT)
                        if (pos[0] < n.size[0]/2) {
                            if (row < n.inputs.length) {
                                helpDOM.selectHelp(n.inputs[row].name)
                            }
                        } else {
                            if (row < n.outputs.length) {
                                helpDOM.selectHelp(n.outputs[row].name)
                            }
                        }
                    } else {
                        //probably widget, but widgets have variable height.
                        let basey = LiteGraph.NODE_SLOT_HEIGHT * inputRows + 6
                        for (let w of n.widgets) {
                            if (w.y) {
                                basey = w.y
                            }
                            let wheight = LiteGraph.NODE_WIDGET_HEIGHT+4
                            if (w.computeSize) {
                                wheight = w.computeSize(n.size[0])[1]
                            }
                            if (pos[1] < basey + wheight) {
                                helpDOM.selectHelp(w.name, w.value)
                                break
                            }
                            basey += wheight
                        }
                    }
                }
            }, 500)
        })
        chainCallback(node, "onMouseLeave", function (e, pos, canvas) {
            if (timeout) {
                clearTimeout(timeout)
                timeout = null
            }
        });
    }
}

function fitHeight(node) {
    node.setSize([node.size[0], node.computeSize([node.size[0], node.size[1]])[1]])
    node?.graph?.setDirtyCanvas(true);
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
            return resp
        } else {
            alert(resp.status + " - " + resp.statusText);
        }
    } catch (error) {
        alert(error);
    }
}
function applyVHSAudioLinksFix(nodeType, nodeData, audio_slot) {
    chainCallback(nodeType.prototype, "onConnectionsChange", function(contype, slot, iscon, linfo) {
        if (contype == LiteGraph.OUTPUT && slot == audio_slot) {
            if (linfo.type == "VHS_AUDIO") {
                this.outputs[audio_slot].type = "AUDIO"
                let tnode = app.graph._nodes_by_id[linfo.target_id]
                let inputDef = LiteGraph.registered_node_types[tnode.type]?.nodeData?.input
                let has_migrated = true
                if (inputDef?.required) {
                    for (let k in inputDef.required) {
                        if (inputDef.required[k][0] == "VHS_AUDIO") {
                            has_migrated = false
                            break
                        }
                    }
                }
                if (has_migrated &&inputDef?.optional) {
                    for (let k in inputDef.optional) {
                        if (inputDef.optional[k][0] == "VHS_AUDIO") {
                            has_migrated = false
                            break
                        }
                    }
                }
                if (!has_migrated) {
                    //need to add node and migrate
                    app.ui.dialog.show("This workflow contains one or more nodes which use the old VHS_AUDIO format. They have been highlighted in red. An AudioToVHSAudio node must be added to convert to this legacy format")
                    tnode.bgcolor = "#C00"
                }
            }
        }
    })
}
function addVAEOutputToggle(nodeType, nodeData) {
    chainCallback(nodeType.prototype, "onConnectionsChange", function(contype, slot, iscon, linfo) {
        let slotType = this.inputs[slot]?.type
        if (contype == LiteGraph.INPUT && slotType == "VAE") {
            if (iscon && linfo) {
                if (this.linkTimeout) {
                    clearTimeout(this.linkTimeout)
                    this.linkTimeout = false
                } else if (this.outputs[0].type == "IMAGE") {
                    this.linkTimeout = setTimeout(() => {
                        if (this.outputs[0].type != "IMAGE") {
                            return
                        }
                        this.linkTimeout = false
                        this.disconnectOutput(0);
                    }, 50)
                }
                this.outputs[0].name = 'LATENT';
                this.outputs[0].type = 'LATENT';
            } else{
                if (this.outputs[0].type == "LATENT") {
                    this.linkTimeout = setTimeout(() => {
                        this.linkTimeout = false
                        this.disconnectOutput(0);
                    }, 50)
                }
                this.outputs[0].name = "IMAGE";
                this.outputs[0].type = "IMAGE";
            }
        }
    });
}
function addVAEInputToggle(nodeType, nodeData) {
    chainCallback(nodeType.prototype, "onConnectionsChange", function(contype, slot, iscon, linf) {
        if (contype == LiteGraph.INPUT && slot == 3 && this.inputs[3].type == "VAE") {
            if (iscon && linf) {
                if (this.linkTimeout) {
                    clearTimeout(this.linkTimeout)
                    this.linkTimeout = false
                } else if (this.inputs[0].type == "IMAGE") {
                    this.linkTimeout = setTimeout(() => {
                        //workaround for out of order loading
                        if (this.inputs[0].type != "IMAGE") {
                            return
                        }
                        this.linkTimeout = false
                        this.disconnectInput(0);
                    }, 50)
                }
                this.inputs[0].type = 'LATENT';
            } else {
                if (this.inputs[0].type == "LATENT") {
                    this.linkTimeout = setTimeout(() => {
                        this.linkTimeout = false
                        this.disconnectInput(0);
                    }, 50)
                }
                this.inputs[0].type = "IMAGE";
            }
        }
    });
}
function cloneType(nodeType, nodeData) {
    nodeData.output[0] = "VHS_DUMMY_NONE"
    chainCallback(nodeType.prototype, "onConnectionsChange", function(contype, slot, iscon, linf) {
        if (contype == LiteGraph.INPUT && slot == 0) {
            let new_type = "VHS_DUMMY_NONE"
            if (iscon && linf) {
                new_type = app.graph.getNodeById(linf.origin_id).outputs[linf.origin_slot].type
            }
            if (this.linkTimeout) {
                clearTimeout(this.linkTimeout)
                this.linkTimeout = false
            }
            this.linkTimeout = setTimeout(() => {
                if (this.outputs[0].type != new_type) {
                    this.outputs[0].type = new_type
                    this.disconnectOutput(0);
                }
                this.linkTimeout = false
            }, 50)
        }
    });
}

function addDateFormatting(nodeType, field, timestamp_widget = false) {
    chainCallback(nodeType.prototype, "onNodeCreated", function() {
        const widget = this.widgets.find((w) => w.name === field);
        widget.serializeValue = () => {
            return applyTextReplacements(app, widget.value);
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
function initializeLoadFormat(nodeType, nodeData) {
    if (!nodeData?.input?.optional?.format) {
        return
    }
    chainCallback(nodeType.prototype, "onNodeCreated", function() {
        let node = this
        let formatWidget = this.widgets.find((w) => w.name === "format")
        formatWidget.options.formats = nodeData.input.optional.format[1].formats
        let base = {}
        for (let widget of this.widgets) {
           if (['force_rate', 'custom_width', 'custom_height',
               'frame_load_cap'].includes(widget.name)) {
               //TODO: filter these options?
               base[widget.name] = widget.options
           }
        }
        chainCallback(formatWidget, "callback", function(value) {
            let format = this.options.formats[value]
            if (!format) {
                return
            }
            if ('target_rate' in format) {
                format.force_rate = {'reset': format.target_rate}
            }
            if ('dim' in format) {
                format.custom_width = {'step': format.dim[0]*10, 'mod': format.dim[1]}
                format.custom_height = {'step': format.dim[0]*10, 'mod': format.dim[1]}
                if (format.dim[2]) {
                    format.custom_width.reset = format.dim[2]
                }
                if (format.dim[3]) {
                    format.custom_height.reset = format.dim[3]
                }
            }
            if ('frames' in format) {
                format.frame_load_cap = {'step': format.frames[0]*10, 'mod': format.frames[1]}
            }
            for (let widget of node.widgets) {
                if (widget.name in base) {
                    let wasDefault = widget.options?.reset == widget.value
                    widget.options = Object.assign({}, base[widget.name], format[widget.name])
                    if (wasDefault && widget.options.reset != undefined) {
                        widget.value = widget.options.reset
                    }
                    widget.callback(widget.value)
                }
            }

        });
        let capWidget = this.widgets.find((w) => w.name === "frame_load_cap")
        let previewWidget = this.widgets.find((w) => w.name === "videopreview")
        chainCallback(previewWidget, "updateSource", () => setTimeout(async () => {
            if (!previewWidget?.value?.params?.filename) {
                return
            }
            let qurl = api.apiURL('/vhs/queryvideo?' + new URLSearchParams(previewWidget.value.params))
            let query = undefined
            try {
                let query_res = await fetch(qurl)
                query = await query_res.json()
            } catch(e) {
                return
            }
            if (!query?.loaded) {
                return
            }
            this.max_frames = query.loaded.frames
        }, 100));
        capWidget.annotation = (value, width) => {
            if (!this.max_frames || value && value < this.max_frames) {
                return
            }
            let format = formatWidget.options.formats[formatWidget.value]
            const div = format?.frames?.[0] ?? 1
            const mod = format?.frames?.[1] ?? 0
            let loadable_frames = this.max_frames
            if ((this.max_frames % div) != mod) {
                loadable_frames = ((this.max_frames - mod)/div|0) * div + mod
            }
            return loadable_frames + "\u21FD"
        }
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
                        if ((await uploadFile(file)).status == 200) {
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
                    if (pathWidget.callback) {
                        pathWidget.callback(path)
                    }
                },
            });
        } else if (type == "video") {
            Object.assign(fileInput, {
                type: "file",
                accept: "video/webm,video/mp4,video/mkv,image/gif",
                style: "display: none",
                onchange: async () => {
                    if (fileInput.files.length) {
                        let resp = await uploadFile(fileInput.files[0])
                        if (resp.status != 200) {
                            //upload failed and file can not be added to options
                            return;
                        }
                        const filename = (await resp.json()).name;
                        pathWidget.options.values.push(filename);
                        pathWidget.value = filename;
                        if (pathWidget.callback) {
                            pathWidget.callback(filename)
                        }
                    }
                },
            });
        } else if (type == "audio") {
            Object.assign(fileInput, {
                type: "file",
                accept: "audio/mpeg,audio/wav,audio/x-wav,audio/ogg",
                style: "display: none",
                onchange: async () => {
                    if (fileInput.files.length) {
                        let resp = await uploadFile(fileInput.files[0])
                        if (resp.status != 200) {
                            //upload failed and file can not be added to options
                            return;
                        }
                        const filename = (await resp.json()).name;
                        pathWidget.options.values.push(filename);
                        pathWidget.value = filename;
                        if (pathWidget.callback) {
                            pathWidget.callback(filename)
                        }
                    }
                },
            });
        }else {
            throw "Unknown upload type"
        }
        document.body.append(fileInput);
        let uploadWidget = this.addWidget("button", "choose " + type + " to upload", "image", () => {
            //clear the active click event
            app.canvas.node_widget = null

            fileInput.click();
        });
        uploadWidget.options.serialize = false;
    });
}

function addVideoPreview(nodeType, isInput=true) {
    chainCallback(nodeType.prototype, "onNodeCreated", function() {
        var element = document.createElement("div");
        const previewNode = this;
        var previewWidget = this.addDOMWidget("videopreview", "preview", element, {
            serialize: false,
            hideOnZoom: false,
            getValue() {
                return element.value;
            },
            setValue(v) {
                element.value = v;
            },
        });
        previewWidget.computeSize = function(width) {
            if (this.aspectRatio && !this.parentEl.hidden) {
                let height = (previewNode.size[0]-20)/ this.aspectRatio + 10;
                if (!(height > 0)) {
                    height = 0;
                }
                this.computedHeight = height + 10;
                return [width, height];
            }
            return [width, -4];//no loaded src, widget should not display
        }
        element.addEventListener('contextmenu', (e)  => {
            e.preventDefault()
            return app.canvas._mousedown_callback(e)
        }, true);
        element.addEventListener('pointerdown', (e)  => {
            e.preventDefault()
            return app.canvas._mousedown_callback(e)
        }, true);
        element.addEventListener('mousewheel', (e)  => {
            e.preventDefault()
            return app.canvas._mousewheel_callback(e)
        }, true);
        previewWidget.value = {hidden: false, paused: false, params: {},
            muted: app.ui.settings.getSettingValue("VHS.DefaultMute")}
        previewWidget.parentEl = document.createElement("div");
        previewWidget.parentEl.className = "vhs_preview";
        previewWidget.parentEl.style['width'] = "100%"
        element.appendChild(previewWidget.parentEl);
        previewWidget.videoEl = document.createElement("video");
        previewWidget.videoEl.controls = false;
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
        previewWidget.videoEl.onmouseenter =  () => {
            previewWidget.videoEl.muted = previewWidget.value.muted
        };
        previewWidget.videoEl.onmouseleave = () => {
            previewWidget.videoEl.muted = true;
        };

        previewWidget.imgEl = document.createElement("img");
        previewWidget.imgEl.style['width'] = "100%"
        previewWidget.imgEl.hidden = true;
        previewWidget.imgEl.onload = () => {
            previewWidget.aspectRatio = previewWidget.imgEl.naturalWidth / previewWidget.imgEl.naturalHeight;
            fitHeight(this);
        };
        previewWidget.parentEl.appendChild(previewWidget.videoEl)
        previewWidget.parentEl.appendChild(previewWidget.imgEl)
        var timeout = null;
        this.updateParameters = (params, force_update) => {
            if (!previewWidget.value.params) {
                if(typeof(previewWidget.value != 'object')) {
                    previewWidget.value =  {hidden: false, paused: false}
                }
                previewWidget.value.params = {}
            }
            Object.assign(previewWidget.value.params, params)
            if (!force_update &&
                app.ui.settings.getSettingValue("VHS.AdvancedPreviews") == 'Never') {
                return;
            }
            if (timeout) {
                clearTimeout(timeout);
            }
            if (force_update) {
                previewWidget.updateSource();
            } else {
                timeout = setTimeout(() => previewWidget.updateSource(),100);
            }
        };
        previewWidget.updateSource = function () {
            if (this.value.params == undefined) {
                return;
            }
            let params =  {}
            let advp = app.ui.settings.getSettingValue("VHS.AdvancedPreviews")
            Object.assign(params, this.value.params);//shallow copy
            params.timestamp = Date.now()
            this.parentEl.hidden = this.value.hidden;
            if (params.format?.split('/')[0] == 'video'
                || advp != 'Never' && (params.format?.split('/')[1] == 'gif')
                || params.format == 'folder') {

                this.videoEl.autoplay = !this.value.paused && !this.value.hidden;
                let target_width = 256
                if (previewWidget.element?.style?.width) {
                    //overscale to allow scrolling. Endpoint won't return higher than native
                    target_width = previewWidget.element.style.width.slice(0,-2)*2;
                }
                let minWidth = app.ui.settings.getSettingValue("VHS.AdvancedPreviewsMinWidth")
                if (target_width < minWidth) {
                    target_width = minWidth
                }
                if (!params.custom_width || !params.custom_height) {
                    params.force_size = target_width+"x?"
                } else {
                    let ar = params.custom_width/params.custom_height
                    params.force_size = target_width+"x"+(target_width/ar)
                }
                params.deadline = app.ui.settings.getSettingValue("VHS.AdvancedPreviewsDeadline")
                if (advp == 'Never' || advp == 'Input Only' && !isInput) {
                    this.videoEl.src = api.apiURL('/view?' + new URLSearchParams(params));
                } else {
                    this.videoEl.src = api.apiURL('/vhs/viewvideo?' + new URLSearchParams(params));
                }
                this.videoEl.hidden = false;
                this.imgEl.hidden = true;
            } else if (params.format?.split('/')[0] == 'image'){
                //Is animated image
                this.imgEl.src = api.apiURL('/view?' + new URLSearchParams(params));
                this.videoEl.hidden = true;
                this.imgEl.hidden = false;
            }
        }
        previewWidget.parentEl.appendChild(previewWidget.videoEl)
        previewWidget.parentEl.appendChild(previewWidget.imgEl)
    });
}
let copiedPath = undefined
function addPreviewOptions(nodeType) {
    chainCallback(nodeType.prototype, "getExtraMenuOptions", function(_, options) {
        // The intended way of appending options is returning a list of extra options,
        // but this isn't used in widgetInputs.js and would require
        // less generalization of chainCallback
        let optNew = []
        const previewWidget = this.widgets.find((w) => w.name === "videopreview");

        let url = null
        if (previewWidget.videoEl?.hidden == false && previewWidget.videoEl.src) {
            if (['input', 'output', 'temp'].includes(previewWidget.value.params.type)) {
                //Use full quality video
                url = api.apiURL('/view?' + new URLSearchParams(previewWidget.value.params));
                //Workaround for 16bit png: Just do first frame
                url = url.replace('%2503d', '001')
            }
        } else if (previewWidget.imgEl?.hidden == false && previewWidget.imgEl.src) {
            url = previewWidget.imgEl.src;
            url = new URL(url);
        }
        if (url) {
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
                        a.setAttribute("download", previewWidget.value.params.filename);
                        document.body.append(a);
                        a.click();
                        requestAnimationFrame(() => a.remove());
                    },
                }
            );
            if (previewWidget.value.params.fullpath) {
                copiedPath = previewWidget.value.params.fullpath
                const blob = new Blob([previewWidget.value.params.fullpath],
                    { type: 'text/plain'})
                optNew.push({
                    content: "Copy output filepath",
                    callback: async () => {
                        await navigator.clipboard.write([
                            new ClipboardItem({
                                'text/plain': blob
                            })])}
                });
            }
            if (previewWidget.value.params.workflow) {
                let wParams = {...previewWidget.value.params,
                    filename: previewWidget.value.params.workflow}
                let wUrl = api.apiURL('/view?' + new URLSearchParams(wParams));
                optNew.push({
                    content: "Save workflow image",
                    callback: () => {
                        const a = document.createElement("a");
                        a.href = wUrl;
                        a.setAttribute("download", previewWidget.value.params.workflow);
                        document.body.append(a);
                        a.click();
                        requestAnimationFrame(() => a.remove());
                    }
                });
            }
        }
        const PauseDesc = (previewWidget.value.paused ? "Resume" : "Pause") + " preview";
        if(previewWidget.videoEl.hidden == false) {
            optNew.push({content: PauseDesc, callback: () => {
                //animated images can't be paused and are more likely to cause performance issues.
                //changing src to a single keyframe is possible,
                //For now, the option is disabled if an animated image is being displayed
                if(previewWidget.value.paused) {
                    previewWidget.videoEl?.play();
                } else {
                    previewWidget.videoEl?.pause();
                }
                previewWidget.value.paused = !previewWidget.value.paused;
            }});
        }
        //TODO: Consider hiding elements if no video preview is available yet.
        //It would reduce confusion at the cost of functionality
        //(if a video preview lags the computer, the user should be able to hide in advance)
        const visDesc = (previewWidget.value.hidden ? "Show" : "Hide") + " preview";
        optNew.push({content: visDesc, callback: () => {
            if (!previewWidget.videoEl.hidden && !previewWidget.value.hidden) {
                previewWidget.videoEl.pause();
            } else if (previewWidget.value.hidden && !previewWidget.videoEl.hidden && !previewWidget.value.paused) {
                previewWidget.videoEl.play();
            }
            previewWidget.value.hidden = !previewWidget.value.hidden;
            previewWidget.parentEl.hidden = previewWidget.value.hidden;
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
        const muteDesc = (previewWidget.value.muted ? "Unmute" : "Mute") + " Preview"
        optNew.push({content: muteDesc, callback: () => {
            previewWidget.value.muted = !previewWidget.value.muted
        }})
        if(options.length > 0 && options[0] != null && optNew.length > 0) {
            optNew.push(null);
        }
        options.unshift(...optNew);
    });
}
function addFormatWidgets(nodeType, nodeData) {
    const formats = nodeData?.input?.required?.format?.[1]?.formats
    function parseFormats(options) {
        options.fullvalues = options._values;
        options._values = [];
        for (let format of options.fullvalues) {
            if (Array.isArray(format)) {
                options._values.push(format[0]);
            } else {
                options._values.push(format);
            }
        }
    }
    chainCallback(nodeType.prototype, "onNodeCreated", function() {
        var formatWidget = null;
        var formatWidgetIndex = -1;
        for(let i = 0; i < this.widgets.length; i++) {
            if (this.widgets[i].name === "format"){
                formatWidget = this.widgets[i];
                formatWidgetIndex = i+1;
            }
        }
        let formatWidgetsCount = 0;
        //Pre-process options to just names
        formatWidget.options._values = formatWidget.options.values;
        parseFormats(formatWidget.options);
        Object.defineProperty(formatWidget.options, "values", {
            set : (value) => {
                formatWidget.options._values  = value;
                parseFormats(formatWidget.options);
            },
            get : () => {
                return formatWidget.options._values;
            }
        })

        formatWidget._value = formatWidget.value;
        Object.defineProperty(formatWidget, "value", {
            set : (value) => {
                formatWidget._value = value;
                let newWidgets = [];
                const fullDef = formatWidget.options.fullvalues.find((w) => Array.isArray(w) ? w[0] === value : w === value);

                if (!Array.isArray(fullDef) && !formats?.[value]) {
                    formatWidget._value = value;
                } else {
                    formatWidget._value = value;
                    let formatWidgets = formats?.[value] ?? fullDef[1]
                    for (let wDef of formatWidgets) {
                        //create widgets. Heavy borrowed from web/scripts/app.js
                        //default implementation doesn't work since it automatically adds
                        //the widget in the wrong spot.
                        //TODO: consider letting this happen and just removing from list?
                        let w = {};
                        w.name = wDef[0];
                        w.config = wDef.slice(1);
                        let inputData = wDef.slice(1);
                        w.type = inputData[0];
                        w.options = inputData[1] ? inputData[1] : {};
                        if (Array.isArray(w.type)) {
                            w.value = w.type[0];
                            w.options.values = w.type;
                            w.type = "combo";
                        }
                        if(inputData[1]?.default != undefined) {
                            w.value = inputData[1].default;
                        }
                        if (w.type == "INT") {
                            Object.assign(w.options, {"precision": 0, "step": 10})
                            w.callback = function (v) {
                                const s = this.options.step / 10;
                                this.value = Math.round(v / s) * s;
                            }
                        }
                        const typeTable = {BOOLEAN: "toggle", STRING: "text", INT: "number", FLOAT: "number"};
                        if (w.type in typeTable) {
                            w.type = typeTable[w.type];
                        }
                        newWidgets.push(w);
                    }
                }
                this.widgets.splice(formatWidgetIndex, formatWidgetsCount, ...newWidgets);
                fitHeight(this);
                formatWidgetsCount = newWidgets.length;
            },
            get : () => {
                return formatWidget._value;
            }
        });
    });
}
function addLoadCommon(nodeType, nodeData) {
    addVideoPreview(nodeType);
    initializeLoadFormat(nodeType, nodeData)
    addPreviewOptions(nodeType);
    chainCallback(nodeType.prototype, "onNodeCreated", function() {
        //widget.callback adds unused arguements which need culling
        const node = this
        function update(key) {
            return (value) => {
                let params = {}
                params[key] = value
                node?.updateParameters(params)
            }
        }
        let prior_ar = -2
        const widthWidget = this.widgets.find((w) => w.name === "custom_width");
        const heightWidget = this.widgets.find((w) => w.name === "custom_height");
        function updateAR(value) {
            let new_ar = -1
            if (widthWidget.value & heightWidget.value) {
                new_ar = widthWidget.value / heightWidget.value
            }
            if (new_ar != prior_ar) {
                node?.updateParameters({'custom_width': widthWidget.value,
                    'custom_height': heightWidget.value})
                prior_ar = new_ar
            }
        }
        const offsetWidget = this.widgets.find((w) => w.name === "start_time");
        if (offsetWidget) {
            offsetWidget.options.step = 10
        }
        let widgetMap = {'frame_load_cap': 'frame_load_cap',
            'skip_first_frames': 'skip_first_frames', 'select_every_nth': 'select_every_nth',
            'start_time': 'start_time', 'force_rate': 'force_rate',
            'custom_width': updateAR, 'custom_height': updateAR,
            'image_load_cap': 'image_load_cap', 'skip_first_images': 'skip_first_images'
        }
        for (let widget of this.widgets) {
            if (widget.name in widgetMap) {
                if (typeof(widgetMap[widget.name]) == 'function') {
                    chainCallback(widget, "callback", widgetMap[widget.name]);
                } else {
                    chainCallback(widget, "callback", update(widgetMap[widget.name]))
                }
            }
            if (widget.type != "button") {
                widget.callback?.(widget.value)
            }
        }
    });
}

function path_stem(path) {
    let i = path.lastIndexOf("/");
    if (i >= 0) {
        return [path.slice(0,i+1),path.slice(i+1)];
    }
    return ["",path];
}
function searchBox(event, [x,y], node) {
    //Ensure only one dialogue shows at a time
    if (this.prompt)
        return;
    this.prompt = true;

    let pathWidget = this;
    let dialog = document.createElement("div");
    dialog.className = "litegraph litesearchbox graphdialog rounded"
    dialog.innerHTML = '<span class="name">Path</span> <input autofocus="" type="text" class="value"><button class="rounded">OK</button><div class="helper"></div>'
    dialog.close = () => {
        dialog.remove();
    }
    document.body.append(dialog);
    if (app.canvas.ds.scale > 1) {
        dialog.style.transform = "scale(" + app.canvas.ds.scale + ")";
    }
    var name_element = dialog.querySelector(".name");
    var input = dialog.querySelector(".value");
    var options_element = dialog.querySelector(".helper");
    input.value = pathWidget.value;

    var timeout = null;
    let last_path = null;
    let extensions = pathWidget.options.vhs_path_extensions

    input.addEventListener("keydown", (e) => {
        dialog.is_modified = true;
        if (e.keyCode == 27) {
            //ESC
            dialog.close();
        } else if (e.keyCode == 13 && e.target.localName != "textarea") {
            pathWidget.value = input.value;
            if (pathWidget.callback) {
                pathWidget.callback(pathWidget.value);
            }
            dialog.close();
        } else {
            if (e.keyCode == 9) {
                //TAB
                input.value = last_path + options_element.firstChild.innerText;
                e.preventDefault();
                e.stopPropagation();
            } else if (e.ctrlKey && (e.keyCode == 87 || e.keyCode == 66)) {
                //Ctrl+w or Ctrl+b
                //most browsers won't support, but it's good QOL for those that do
                input.value = path_stem(input.value.slice(0,-1))[0]
                e.preventDefault();
                e.stopPropagation();
            } else if (e.ctrlKey && e.keyCode == 71) {
                //Ctrl+g
                //Temporarily disables extension filtering to show all files
                e.preventDefault();
                e.stopPropagation();
                extensions = undefined
                last_path = null;
            }
            if (timeout) {
                clearTimeout(timeout);
            }
            timeout = setTimeout(updateOptions, 10);
            return;
        }
        this.prompt=false;
        e.preventDefault();
        e.stopPropagation();
    });

    var button = dialog.querySelector("button");
    button.addEventListener("click", (e) => {
        pathWidget.value = input.value;
        if (pathWidget.callback) {
            pathWidget.callback(pathWidget.value);
        }
        //unsure why dirty is set here, but not on enter-key above
        node.graph.setDirtyCanvas(true);
        dialog.close();
        this.prompt = false;
    });
    var rect = app.canvas.canvas.getBoundingClientRect();
    var offsetx = -20;
    var offsety = -20;
    if (rect) {
        offsetx -= rect.left;
        offsety -= rect.top;
    }

    if (event) {
        dialog.style.left = event.clientX + offsetx + "px";
        dialog.style.top = event.clientY + offsety + "px";
    } else {
        dialog.style.left = canvas.width * 0.5 + offsetx + "px";
        dialog.style.top = canvas.height * 0.5 + offsety + "px";
    }
    //Search code
    let options = []
    function addResult(name, isDir) {
        let el = document.createElement("div");
        el.innerText = name;
        el.className = "litegraph lite-search-item";
        if (isDir) {
            el.className += " is-dir";
            el.addEventListener("click", (e) => {
                input.value = last_path+name
                if (timeout) {
                    clearTimeout(timeout);
                }
            timeout = setTimeout(updateOptions, 10);
            });
        } else {
            el.addEventListener("click", (e) => {
                pathWidget.value = last_path+name;
                if (pathWidget.callback) {
                    pathWidget.callback(pathWidget.value);
                }
                dialog.close();
                pathWidget.prompt = false;
            });
        }
        options_element.appendChild(el);
    }
    async function updateOptions() {
        timeout = null;
        let [path, remainder] = path_stem(input.value);
        if (last_path != path) {
            //fetch options.  Must block execution here, so update should be async?
            let params = {path : path}
            if (extensions) {
                params.extensions = extensions
            }
            let optionsURL = api.apiURL('/vhs/getpath?' + new URLSearchParams(params));
            try {
                let resp = await fetch(optionsURL);
                options = await resp.json();
                options = options.map((o) => o.replace('.','\0'))
                options = options.sort()
                options = options.map((o) => o.replace('\0','.'))
            } catch(e) {
                options = []
            }
            last_path = path;
        }
        options_element.innerHTML = '';
        //filter options based on remainder
        for (let option of options) {
            if (option.startsWith(remainder)) {
                let isDir = option.endsWith('/')
                addResult(option, isDir);
            }
        }
    }

    setTimeout(async function() {
        input.focus();
        await updateOptions();
    }, 10);

    return dialog;
}
function button_action(widget) {
  if (
    widget.options?.reset == undefined &&
    widget.options?.disable == undefined
  ) {
    return 'None'
  }
  if (
    widget.options.reset != undefined &&
    widget.value != widget.options.reset
  ) {
    return 'Reset'
  }
  if (
    widget.options.disable != undefined &&
    widget.value != widget.options.disable
  ) {
    return 'Disable'
  }
  if (widget.options.reset != undefined) {
    return 'No Reset'
  }
  return 'No Disable'
}
function inner_value_change(widget, value, node, pos) {
  widget.value = value
  if (widget.options?.property && widget.options.property in node.properties) {
    node.setProperty(widget.options.property, value)
  }
  if (widget.callback) {
    widget.callback(widget.value, app.canvas, node, event)
  }
}
function drawAnnotated(ctx, node, widget_width, y, H) {
  const litegraph_base = LiteGraph
  const show_text = app.canvas.ds.scale > 0.5
  const margin = 15
  ctx.textAlign = 'left'
  ctx.strokeStyle = litegraph_base.WIDGET_OUTLINE_COLOR
  ctx.fillStyle = litegraph_base.WIDGET_BGCOLOR
  ctx.beginPath()
  if (show_text)
    ctx.roundRect(margin, y, widget_width - margin * 2, H, [H * 0.5])
  else ctx.rect(margin, y, widget_width - margin * 2, H)
  ctx.fill()
  if (show_text) {
    const monospace_font = ctx.font.split(' ')[0] + ' monospace'
    if (!this.disabled) ctx.stroke()
    const button = button_action(this)
    if (button != 'None') {
      ctx.save()
      ctx.font = monospace_font
      if (button.startsWith('No ')) {
        ctx.fillStyle = litegraph_base.WIDGET_OUTLINE_COLOR
      } else {
        ctx.fillStyle = litegraph_base.WIDGET_TEXT_COLOR
      }
      if (button.endsWith('Reset')) {
        ctx.fillText('\u21ba', widget_width - margin - 33, y + H * 0.7)
      } else {
        ctx.fillText('\u2298', widget_width - margin - 33, y + H * 0.7)
      }
      ctx.restore()
    }
    ctx.fillStyle = litegraph_base.WIDGET_TEXT_COLOR
    if (!this.disabled) {
      ctx.beginPath()
      ctx.moveTo(margin + 16, y + 5)
      ctx.lineTo(margin + 6, y + H * 0.5)
      ctx.lineTo(margin + 16, y + H - 5)
      ctx.fill()
      ctx.beginPath()
      ctx.moveTo(widget_width - margin - 16, y + 5)
      ctx.lineTo(widget_width - margin - 6, y + H * 0.5)
      ctx.lineTo(widget_width - margin - 16, y + H - 5)
      ctx.fill()
    }
    ctx.fillStyle = litegraph_base.WIDGET_SECONDARY_TEXT_COLOR
    ctx.fillText(this.label || this.name, margin * 2 + 5, y + H * 0.7)
    ctx.fillStyle = litegraph_base.WIDGET_TEXT_COLOR
    ctx.textAlign = 'right'
    const text = Number(this.value).toFixed(
      this.options.precision !== undefined ? this.options.precision : 3
    )
    let value_offset = margin * 2 + 20
    if (this.options.unit) {
      ctx.save()
      ctx.font = monospace_font
      ctx.fillStyle = litegraph_base.WIDGET_OUTLINE_COLOR
      ctx.fillText(this.options.unit, widget_width - value_offset, y + H * 0.7)
      value_offset += ctx.measureText(this.options.unit).width
      ctx.restore()
    }
    ctx.fillText(text, widget_width - value_offset, y + H * 0.7)

    const value_width = ctx.measureText(text).width
    const name_width = ctx.measureText(this.label || this.name).width
    const free_width =
      widget_width - (value_width + name_width + value_offset + 40)

    let annotation = ''
    if (this.annotation) {
      annotation = this.annotation(this.value, free_width)
    } else if (
      this.options.annotation &&
      this.value in this.options.annotation
    ) {
      annotation = this.options.annotation[this.value]
    }
    if (annotation) {
      ctx.fillStyle = litegraph_base.WIDGET_OUTLINE_COLOR
      const annotation_width = ctx.measureText(annotation).width
      if (free_width < annotation_width) {
        //Enforcing a widget's requested minimum width seems ill supported
        //hiding annotation is best, but existence should still be indicated
        annotation = '…'
      }
      ctx.fillText(
        annotation,
        widget_width - 5 - value_width - value_offset,
        y + H * 0.7
      )
    }
  }
}
function mouseAnnotated(event, [x, y], node) {
  const button = button_action(this)
  const widget_width = this.width || node.size[0]
  const old_value = this.value
  const delta = x < 40 ? -1 : x > widget_width - 48 ? 1 : 0
  const margin = 15
  var allow_scroll = true
  if (delta) {
    if (x > -3 && x < widget_width + 3) {
      allow_scroll = false
    }
  }
  if (allow_scroll && event.type == 'pointermove') {
    if (event.deltaX)
      this.value += event.deltaX * 0.1 * (this.options.step || 1)
    if (this.options.min != null && this.value < this.options.min) {
      this.value = this.options.min
    }
    if (this.options.max != null && this.value > this.options.max) {
      this.value = this.options.max
    }
  } else if (event.type == 'pointerdown') {
    if (x > widget_width - margin - 34 && x < widget_width - margin - 18) {
      if (button == 'Reset') {
        this.value = this.options.reset
      } else if (button == 'Disable') {
        this.value = this.options.disable
      }
    } else {
      this.value += delta * 0.1 * (this.options.step || 1)
      if (this.options.min != null && this.value < this.options.min) {
        this.value = this.options.min
      }
      if (this.options.max != null && this.value > this.options.max) {
        this.value = this.options.max
      }
    }
  } //end mousedown
  else if (event.type == 'pointerup') {
    if (event.click_time < 200 && delta == 0) {
      app.canvas.prompt(
        'Value',
        this.value,
        function (v) {
          //NOTE: Original code uses eval here. This will not be reproduced
          this.value = Number(v)
          inner_value_change(this, this.value, node, [x, y])
        }.bind(this),
        event
      )
    }
  }

  if (old_value != this.value)
    setTimeout(
      function () {
        inner_value_change(this, this.value, node, [x, y])
      }.bind(this),
      20
    )
  return true
}
function makeAnnotated(widget, inputData) {
    const callback_orig = widget.callback
    Object.assign(widget, {
        type: "BOOLEAN",//Horrific, not namespaced, nonsensical, easier than upstreaming
        draw: drawAnnotated,
        mouse: mouseAnnotated,
        computeSize(width) {
            return [width, 20]
        },
        callback(v) {
            if (v == 0) {
                return
            }
            if (this.options?.mod == undefined) {
                return callback_orig.apply(this, arguments);
            }
            const s = this.options.step / 10
            let sh = this.options.mod
            this.value = Math.round((v - sh) / s) * s + sh
        },
        config: inputData,
        options: Object.assign({},  inputData[1], widget.options)
    })
    return widget
}
let latentPreviewNodes = new Set()
app.registerExtension({
    name: "VideoHelperSuite.Core",
    settings: [
      {
        id: 'VHS.AdvancedPreviews',
        category: ['🎥🅥🅗🅢', 'Previews', 'Advanced Previews'],
        name: 'Advanced Previews',
        tooltip: 'Automatically transcode previews on request. Required for advanced functionality',
        type: 'combo',
        options: ['Never', 'Always', 'Input Only'],
        defaultValue: 'Input Only',
      },
      {
        id: 'VHS.AdvancedPreviewsMinWidth',
        category: ['🎥🅥🅗🅢', 'Previews', 'Min Width'],
        name: 'Minimum preview width',
        tooltip: 'Advanced previews have their resolution downscaled to the node size for performance. While a node can be resized to increase preview quality, a minimum width can be set that previews won\'t be downscaled beneath. Preveiws will never be upscaled, so this can safely be set large.',
        type: 'number',
        attrs: {
          min: 0,
          step: 1,
          max: 3840,
        },
        defaultValue: 0,
      },
      {
        id: 'VHS.AdvancedPreviewsDeadline',
        category: ['🎥🅥🅗🅢', 'Previews', 'Deadline'],
        name: 'Deadline',
        tooltip: 'Determines how much time can be spent when encoding advanced previews. Realtime results in reduced quality, but good will likely cause the preview to stutter as initial generation occurs',
        type: 'combo',
        options: ['realtime', 'good'],
        defaultValue: 'realtime',
      },
      {
        id: 'VHS.AdvancedPreviewsDefaultMute',
        category: ['🎥🅥🅗🅢', 'Previews', 'Default Mute'],
        name: 'Mute videos by default',
        type: 'boolean',
        defaultValue: false,
      },
      {
        id: 'VHS.LatentPreview',
        category: ['🎥🅥🅗🅢', 'Sampling', 'Latent Previews'],
        name: 'Display animated previews when sampling',
        type: 'boolean',
        defaultValue: false,
        onChange(value) {
            if (!value) {
                //Remove any previewWidgets
                for (let n of latentPreviewNodes) {
                    let i = n?.widgets?.findIndex((w) => w.name == 'vhslatentpreview')
                    if (i >= 0) {
                        n.widgets.splice(i,1)[0].onRemove()
                    }
                }
                latentPreviewNodes = new Set()
            }
        },
      },
      {
        id: "VHS.LatentPreviewRate",
        category: ['🎥🅥🅗🅢', 'Sampling', 'Latent Preview Rate'],
        name: "Playback rate override.",
        type: 'number',
        attrs: {
          min: 0,
          step: 1,
          max: 60
        },
        tooltip:
          'Force a specific frame rate for the playback of latent frames. This should not be confused with the output frame rate and will not match for video models.',
        defaultValue: 0,
      },
      {
        id: 'VHS.MetadataImage',
        category: ['🎥🅥🅗🅢', 'Output', 'MetadataImage'],
        name: 'Save png of first frame for metadata',
        type: 'boolean',
        defaultValue: true,
      },
      {
        id: 'VHS.KeepIntermediate',
        category: ['🎥🅥🅗🅢', 'Output', 'Keep Intermediate'],
        name: 'Keep required intermediate files after sucessful execution',
        type: 'boolean',
        defaultValue: true,
      },
    ],

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if(nodeData?.name?.startsWith("VHS_")) {
            useKVState(nodeType);
            if (nodeData.description) {
                let description = nodeData.description
                let el = document.createElement("div")
                el.innerHTML = description
                if (!el.children.length) {
                    //Is plaintext. Do minor convenience formatting
                    let chunks = description.split('\n')
                    nodeData.description = chunks[0]
                    description = chunks.join('<br>')
                } else {
                    nodeData.description = el.querySelector('#VHS_shortdesc')?.innerHTML || el.children[1]?.firstChild?.innerHTML
                }
                chainCallback(nodeType.prototype, "onNodeCreated", function () {
                    helpDOM.addHelp(this, nodeType, description)
                    this.setSize(this.computeSize())
                })
            }
            chainCallback(nodeType.prototype, "onNodeCreated", function () {
                let new_widgets = []
                if (this.widgets) {
                    for (let w of this.widgets) {
                        let input = this.constructor.nodeData.input
                        let config = input?.required[w.name] ?? input.optional[w.name]
                        if (!config) {
                            continue
                        }
                        if (w?.type == "text" && config[1].vhs_path_extensions) {
                            new_widgets.push(app.widgets.VHSPATH({}, w.name, ["VHSPATH", config[1]]));
                        } else if (w?.type == "number") {
                            new_widgets.push(makeAnnotated(w, config))
                        } else {
                            new_widgets.push(w)
                        }
                    }
                    this.widgets = new_widgets;
                }
                const originalAddInput = this.addInput;
                this.addInput = function(name, type, options) {
                    if (options.widget) {
                        //Is Converted Widget
                        const widget = this.widgets.find((w) => w.name == name)
                        if (widget?.config) {
                            //Has override for type
                            type = widget.config[0]
                            if (type == 'FLOAT') {
                                type = "FLOAT,INT"
                            }
                            const symbol = Object.getOwnPropertySymbols(options.widget)[0]
                            options.widget[symbol] = () => widget.config
                        }
                    }
                    return originalAddInput.apply(this, [name, type, options])
                }
            });
        }
        if (nodeData?.name == "VHS_LoadImages") {
            addUploadWidget(nodeType, nodeData, "directory", "folder");
            chainCallback(nodeType.prototype, "onNodeCreated", function() {
                const pathWidget = this.widgets.find((w) => w.name === "directory");
                chainCallback(pathWidget, "callback", (value) => {
                    if (!value) {
                        return;
                    }
                    let params = {filename : value, type : "input", format: "folder"};
                    this.updateParameters(params, true);
                });
            });
            addLoadCommon(nodeType, nodeData);
        } else if (nodeData?.name == "VHS_LoadImagesPath") {
            addUploadWidget(nodeType, nodeData, "directory", "folder");
            chainCallback(nodeType.prototype, "onNodeCreated", function() {
                const pathWidget = this.widgets.find((w) => w.name === "directory");
                chainCallback(pathWidget, "callback", (value) => {
                    if (!value) {
                        return;
                    }
                    let params = {filename : value, type : "path", format: "folder"};
                    this.updateParameters(params, true);
                });
            });
            addLoadCommon(nodeType, nodeData);
        } else if (nodeData?.name == "VHS_LoadVideo" || nodeData?.name == "VHS_LoadVideoFFmpeg") {
            chainCallback(nodeType.prototype, "onNodeCreated", function() {
                const pathWidget = this.widgets.find((w) => w.name === "video");
                chainCallback(pathWidget, "callback", (value) => {
                    if (!value) {
                        return;
                    }
                    let parts = ["input", value];
                    let extension_index = parts[1].lastIndexOf(".");
                    let extension = parts[1].slice(extension_index+1);
                    let format = "video"
                    if (["gif", "webp", "avif"].includes(extension)) {
                        format = "image"
                    }
                    format += "/" + extension;
                    let params = {filename : parts[1], type : parts[0], format: format};
                    this.updateParameters(params, true);
                });
            });
            addUploadWidget(nodeType, nodeData, "video");
            addLoadCommon(nodeType, nodeData);
            addVAEOutputToggle(nodeType, nodeData);
            applyVHSAudioLinksFix(nodeType, nodeData, 2)
        } else if (nodeData?.name == "VHS_LoadAudioUpload") {
            addUploadWidget(nodeType, nodeData, "audio", "audio");
            applyVHSAudioLinksFix(nodeType, nodeData, 0)
        } else if (nodeData?.name == "VHS_LoadAudio"){
            applyVHSAudioLinksFix(nodeType, nodeData, 0)
        } else if (nodeData?.name == "VHS_LoadVideoPath" || nodeData?.name == "VHS_LoadVideoFFmpegPath") {
            chainCallback(nodeType.prototype, "onNodeCreated", function() {
                const pathWidget = this.widgets.find((w) => w.name === "video");
                chainCallback(pathWidget, "callback", (value) => {
                    let extension_index = value.lastIndexOf(".");
                    let extension = value.slice(extension_index+1);
                    let format = "video"
                    if (["gif", "webp", "avif"].includes(extension)) {
                        format = "image"
                    }
                    format += "/" + extension;
                    let params = {filename : value, type: "path", format: format};
                    this.updateParameters(params, true);
                });
            });
            addLoadCommon(nodeType, nodeData);
            addVAEOutputToggle(nodeType, nodeData);
            applyVHSAudioLinksFix(nodeType, nodeData, 2)
        } else if (nodeData?.name == "VHS_LoadImagePath") {
            addLoadCommon(nodeType, nodeData);
            addVAEOutputToggle(nodeType, nodeData);
            chainCallback(nodeType.prototype, "onNodeCreated", function() {
                const pathWidget = this.widgets.find((w) => w.name === "image");
                chainCallback(pathWidget, "callback", (value) => {
                    let extension_index = value.lastIndexOf(".");
                    let extension = value.slice(extension_index+1);
                    let format = "video" +  "/" + extension;
                    let params = {filename : value, type: "path", format: format};
                    this.updateParameters(params, true);
                });
            });
        } else if (nodeData?.name == "VHS_VideoCombine") {
            addDateFormatting(nodeType, "filename_prefix");
            chainCallback(nodeType.prototype, "onExecuted", function(message) {
                if (message?.gifs) {
                    this.updateParameters(message.gifs[0], true);
                }
            });
            addVideoPreview(nodeType, false);
            addPreviewOptions(nodeType);
            addFormatWidgets(nodeType, nodeData);
            addVAEInputToggle(nodeType, nodeData)

            chainCallback(nodeType.prototype, "onNodeCreated", function() {
                this._outputs = this.outputs
                Object.defineProperty(this, "outputs", {
                    set : function(value) {
                        this._outputs = value;
                        requestAnimationFrame(() => {
                            if (app.nodeOutputs[this.id + ""]) {
                                this.updateParameters(app.nodeOutputs[this.id+""].gifs[0], true);
                            }
                        })
                    },
                    get : function() {
                        return this._outputs;
                    }
                });
                //Display previews after reload/ loading workflow
                this.updateParameters({}, true);
            });
        } else if (nodeData?.name == "VHS_SaveImageSequence") {
            //Disabled for safety as VHS_SaveImageSequence is not currently merged
            //addDateFormating(nodeType, "directory_name", timestamp_widget=true);
            //addTimestampWidget(nodeType, nodeData, "directory_name")
        } else if (nodeData?.name == "VHS_BatchManager") {
            chainCallback(nodeType.prototype, "onNodeCreated", function() {
                this.widgets.push({name: "count", type: "dummy", value: 0,
                    computeSize: () => {return [0,-4]},
                    afterQueued: function() {this.value++;}});
            });
        } else if (nodeData?.name == "VHS_Unbatch") {
            cloneType(nodeType, nodeData)
        }
    },
    async getCustomWidgets() {
        return {
            VHSPATH(node, inputName, inputData) {
                let w = {
                    name : inputName,
                    type : "VHS.PATH",
                    value : "",
                    draw : function(ctx, node, widget_width, y, H) {
                        //Adapted from litegraph.core.js:drawNodeWidgets
                        var show_text = app.canvas.ds.scale > 0.5;
                        var margin = 15;
                        var text_color = LiteGraph.WIDGET_TEXT_COLOR;
                        var secondary_text_color = LiteGraph.WIDGET_SECONDARY_TEXT_COLOR;
                        ctx.textAlign = "left";
                        ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR;
                        ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR;
                        ctx.beginPath();
                        if (show_text)
                            ctx.roundRect(margin, y, widget_width - margin * 2, H, [H * 0.5]);
                        else
                            ctx.rect( margin, y, widget_width - margin * 2, H );
                        ctx.fill();
                        if (show_text) {
                            if(!this.disabled)
                                ctx.stroke();
                            ctx.save();
                            ctx.beginPath();
                            ctx.rect(margin, y, widget_width - margin * 2, H);
                            ctx.clip();

                            //ctx.stroke();
                            ctx.fillStyle = secondary_text_color;
                            const label = this.label || this.name;
                            if (label != null) {
                                ctx.fillText(label, margin * 2, y + H * 0.7);
                            }
                            ctx.fillStyle = this.value ? text_color : '#777';
                            ctx.textAlign = "right";
                            let disp_text = this.format_path(String(this.value || this.options.placeholder))
                            ctx.fillText(disp_text, widget_width - margin * 2, y + H * 0.7); //30 chars max
                            ctx.restore();
                        }
                    },
                    mouse : searchBox,
                    options : {},
                    format_path : function(path) {
                        //Formats the full path to be under 30 characters
                        if (path.length <= 30) {
                            return path;
                        }
                        let filename = path_stem(path)[1]
                        if (filename.length > 28) {
                            //may all fit, but can't squeeze more info
                            return filename.substr(0,30);
                        }
                        //TODO: find solution for windows, path[1] == ':'?
                        let isAbs = path[0] == '/';
                        let partial = path.substr(path.length - (isAbs ? 28:29))
                        let cutoff = partial.indexOf('/');
                        if (cutoff < 0) {
                            //Can occur, but there isn't a nicer way to format
                            return path.substr(path.length-30);
                        }
                        return (isAbs ? '/…':'…') + partial.substr(cutoff);

                    }
                };
                if (inputData.length > 1) {
                    w.options = inputData[1]
                    if (inputData[1].default) {
                        w.value = inputData[1].default;
                    }
                }

                if (!node.widgets) {
                    node.widgets = [];
                }
                node.widgets.push(w);
                return w;
            },
            VHSFLOAT(node, inputName, inputData, app) {
                let w = app.widgets.FLOAT(node, inputName, inputData, app)
                return makeAnnotated(w, inputData);
            },
            VHSINT(node, inputName, inputData, app_arg) {
                let w = app.widgets.INT(node, inputName, inputData, app)
                return makeAnnotated(w, inputData);
            }
        }
    },
    async loadedGraphNode(node) {
        //Check and migrate inputs named batch_manager from old workflows
        if (node.type?.startsWith("VHS_") && node.inputs) {
            const batchInput = node.inputs.find((i) => i.name == "batch_manager")
            if (batchInput) {
                batchInput.name = "meta_batch"
            }
        }
    },
    async beforeConfigureGraph(graphData, missingNodeTypes) {
        if(helpDOM?.node) {
            helpDOM.node = undefined
        }
    },
    async setup() {
        //cg-use-everywhere link workaround
        //particularly invasive, plan to remove
        let originalGraphToPrompt = app.graphToPrompt
        let graphToPrompt = async function() {
            let res = await originalGraphToPrompt.apply(this, arguments);
            for (let n of app.graph._nodes) {
                if (n?.type?.startsWith('VHS_LoadVideo')) {
                    if (!n?.inputs[1]?.link && res?.output[n.id]?.inputs?.vae) {
                        delete res.output[n.id].inputs.vae
                    }
                }
            }
            res.workflow.extra['VHS_latentpreview'] = app.ui.settings.getSettingValue("VHS.LatentPreview")
            res.workflow.extra['VHS_latentpreviewrate'] = app.ui.settings.getSettingValue("VHS.LatentPreviewRate")
            res.workflow.extra['VHS_MetadataImage'] = app.ui.settings.getSettingValue("VHS.MetadataImage")
            res.workflow.extra['VHS_KeepIntermediate'] = app.ui.settings.getSettingValue("VHS.KeepIntermediate")
            return res
        }
        app.graphToPrompt = graphToPrompt
        //Add a handler for pasting video data
        document.addEventListener('paste', async (e) => {
            if (!e.target.classList.contains('litegraph') &&
                !e.target.classList.contains('graph-canvas-container')) {
                    return
                }
            let data = e.clipboardData || window.clipboardData
            let filepath = data.getData('text/plain')
            let video
            for (const item of data.items) {
                if (item.type.startsWith('video/')) {
                    video = item
                    break
                }
            }
            if (filepath && copiedPath == filepath) {
                //Add a Load Video (Path) and populate filepath
                const pastedNode = LiteGraph.createNode('VHS_LoadVideoPath')
                app.graph.add(pastedNode)
                pastedNode.pos[0] = app.canvas.graph_mouse[0]
                pastedNode.pos[1] = app.canvas.graph_mouse[1]
                pastedNode.widgets[0].value = filepath
                pastedNode.widgets[0].callback?.(filepath)
            } else if (video && false) {
                //Disabled due to lack of testing
                //Add a Load Video (Upload), then upload the file, then select the file
                const pastedNode = LiteGraph.createNode('VHS_LoadVideo')
                app.graph.add(pastedNode)
                pastedNode.pos[0] = app.canvas.graph_mouse[0]
                pastedNode.pos[1] = app.canvas.graph_mouse[1]
                const pathWidget = pastedNode.widgets[0]
                //TODO: upload to pasted dir?
                const blob = video.getAsFile()
                const resp = await uploadFile(blob)
                if (resp.status != 200) {
                    //upload failed and file can not be added to options
                    return;
                }
                const filename = (await resp.json()).name;
                pathWidget.options.values.push(filename);
                pathWidget.value = filename;
                pathWidget.callback?.(filename)
            } else {
                return
            }
            e.preventDefault()
            e.stopImmediatePropagation()
            return false
        }, true)
    },
    async init() {
        if (app.ui.settings.getSettingValue("VHS.AdvancedPreviews") == true) {
            app.ui.settings.setSettingValue("VHS.AdvancedPreviews", 'Always')
        }
        if (app.ui.settings.getSettingValue("VHS.AdvancedPreviews") == false) {
            app.ui.settings.setSettingValue("VHS.AdvancedPreviews", 'Never')
        }
        if (app.VHSHelp != helpDOM) {
            helpDOM = app.VHSHelp
        } else {
            initHelpDOM()
        }
        let e = app.extensions.filter((w) => w.name == 'UVR5.AudioPreviewer')
        if (e.length) {
            let orig = e[0].beforeRegisterNodeDef
            e[0].beforeRegisterNodeDef = function(nodeType, nodeData, app) {
                if(!nodeData?.name?.startsWith("VHS_")) {
                    return orig.apply(this, arguments);
                }
            }
        }
    },
});
let previewImages = []
let animateInterval
api.addEventListener('VHS_latentpreview', ({ detail }) => {
    let setting = app.ui.settings.getSettingValue("VHS.LatentPreview")
    if (!setting) {
        return
    }
    let id = app.runningNodeId
    if (id == null) {
        return
    }
    let previewNode = app.graph.getNodeById(id)
    latentPreviewNodes.add(previewNode)
    let previewWidget = previewNode.widgets.find((w) => w.name == "vhslatentpreview")
    if (!previewWidget) {
        let canvasEl = document.createElement("canvas")
        previewWidget = previewNode.addDOMWidget("vhslatentpreview", "vhscanvas", canvasEl, {
            serialize: false,
            hideOnZoom: false,
        });
        previewWidget.computeSize = function(width) {
            if (this.aspectRatio) {
                let height = (previewNode.size[0]-20)/ this.aspectRatio + 10;
                if (!(height > 0)) {
                    height = 0;
                }
                this.computedHeight = height + 10;
                return [width, height];
            }
            return [width, -4];//no loaded src, widget should not display
        }
    }
    let firstPreview = true
    let ctx
    previewImages = []
    previewImages.length = detail.length
    let displayIndex = 0
    if (animateInterval) {
        clearTimeout(animateInterval)
    }
    animateInterval = setInterval(() => {
        if (app.runningNodeId != id) {
            clearTimeout(animateInterval)
            animateInterval = undefined
            return
        }
        if (!previewImages[displayIndex]) {
            return
        }
        let canvasEl = previewWidget.element
        if (!ctx) {
            previewWidget.aspectRatio = previewImages[displayIndex].width / previewImages[displayIndex].height
            canvasEl.width = previewImages[displayIndex].width
            canvasEl.height = previewImages[displayIndex].height
            ctx = canvasEl.getContext("2d")
            fitHeight(previewNode)
        }
        ctx.drawImage(previewImages[displayIndex],0,0)
        displayIndex = (displayIndex + 1) % previewImages.length
    }, 1000/detail.rate);
});
api.addEventListener('b_preview', async (e) => {
    if (!animateInterval) {
        return
    }
    e.preventDefault()
    e.stopImmediatePropagation()
    e.stopPropagation()
    const ab = await e.detail.slice(0,8).arrayBuffer()
    const index = new DataView(ab).getUint32(4)
    previewImages[index] = await window.createImageBitmap(e.detail.slice(8))
    return false
}, true);
