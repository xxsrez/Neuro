// Package assets embeds the portable report so a compiled binary can run
// from any working directory without looking for a template on disk.
package assets

import _ "embed"

//go:embed report.html
var ReportTemplate string
