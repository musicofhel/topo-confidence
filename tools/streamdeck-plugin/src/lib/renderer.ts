export type DaemonState = "active" | "idle" | "exhausted" | "stopped" | "no_conn";

type ButtonColor = "green" | "yellow" | "red";

const PHASE_LABELS: Record<string, string> = {
	triage: "TRIAGE",
	scriptgen: "SCRIPT\nGEN",
	experiment: "EXPERI\nMENT",
};

const SIZE = 144;
const CX = SIZE / 2;
const CY = SIZE / 2;

function colorToSvg(color: ButtonColor, flashOn: boolean): { bg: string; text: string } {
	switch (color) {
		case "green":
			return { bg: "#16a34a", text: "#ffffff" };
		case "yellow":
			return { bg: flashOn ? "#eab308" : "#a16207", text: "#000000" };
		case "red":
			return { bg: "#dc2626", text: "#ffffff" };
	}
}

function renderTwoLine(line1: string, line2: string, bg: string, textColor: string): string {
	return `<svg xmlns="http://www.w3.org/2000/svg" width="${SIZE}" height="${SIZE}">
<rect width="${SIZE}" height="${SIZE}" rx="12" fill="${bg}"/>
<text x="${CX}" y="${CY - 4}" text-anchor="middle" font-family="Arial,sans-serif" font-size="28" font-weight="bold" fill="${textColor}">${line1}</text>
<text x="${CX}" y="${CY + 28}" text-anchor="middle" font-family="Arial,sans-serif" font-size="28" font-weight="bold" fill="${textColor}">${line2}</text>
</svg>`;
}

function renderOneLine(text: string, bg: string, textColor: string): string {
	return `<svg xmlns="http://www.w3.org/2000/svg" width="${SIZE}" height="${SIZE}">
<rect width="${SIZE}" height="${SIZE}" rx="12" fill="${bg}"/>
<text x="${CX}" y="${CY + 10}" text-anchor="middle" font-family="Arial,sans-serif" font-size="28" font-weight="bold" fill="${textColor}">${text}</text>
</svg>`;
}

export function renderSvg(opts: {
	state: DaemonState;
	phase: string;
	flashOn: boolean;
}): string {
	const label = PHASE_LABELS[opts.phase] ?? opts.phase.toUpperCase();
	let color: ButtonColor;
	switch (opts.state) {
		case "active":
			color = "yellow";
			break;
		case "idle":
		case "exhausted":
			color = "green";
			break;
		case "stopped":
		case "no_conn":
			color = "red";
			break;
	}
	const { bg, text: textColor } = colorToSvg(color, opts.flashOn);
	const lines = label.split("\n");
	if (lines.length === 1) {
		return renderOneLine(lines[0], bg, textColor);
	}
	return renderTwoLine(lines[0], lines[1], bg, textColor);
}

export function renderMonitorSvg(opts: {
	title: string;
	subtitle: string;
	color: ButtonColor;
	flashOn: boolean;
}): string {
	const { bg, text: textColor } = colorToSvg(opts.color, opts.flashOn);
	return renderTwoLine(opts.title, opts.subtitle, bg, textColor);
}

export function svgToDataUri(svg: string): string {
	return `data:image/svg+xml,${encodeURIComponent(svg)}`;
}
