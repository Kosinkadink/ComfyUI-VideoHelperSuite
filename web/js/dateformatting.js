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
    },
});
