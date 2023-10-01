import { app } from '../../../scripts/app.js'

// Simple date formatter
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

function formatDate(text, date) {
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

app.registerExtension({
	name: "VideoHelperSuite.DateFormatting",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData?.name == "VHS_VideoCombine") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

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

				return r;
			};
		}
		if (nodeData?.name == "VHS_SaveImageSequence") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

				const directoryWidget = this.widgets.find((w) => w.name === "directory_name");
				const timestampWidget = this.widgets.find((w) => w.name === "timestamp_directory");
                directoryWidget.serializeValue = () => {
                    if (timestampWidget.value) {
                        //ignore actual value and return timestamp
                        return formatDate("yyyy-MM-ddThh:mm:ss", new Date());
                    }
                    return directoryWidget.value
                };
                directoryWidget._value = directoryWidget.value;
                Object.defineProperty(directoryWidget, "value", {
                    set : function(value) {
                        directoryWidget._value = value;
                    },
                    get : function() {
                        if (timestampWidget.value) {
                            return "yyyy-MM-ddThh:mm:ss";
                        }
                        return directoryWidget._value;
                    }
                });

				return r;
			};
		}
    },
});
