package logger

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/fatih/color"
	yamlconv "github.com/ghodss/yaml"
	"github.com/golang/protobuf/jsonpb"
	"github.com/golang/protobuf/proto"
	yaml "gopkg.in/yaml.v2"
)

// Logger creates a new instance of a logger.
type Logger func(format string, a ...interface{})

const (
	// ErrorLevel logging.
	ErrorLevel = 1
	// WarningLevel logging.
	WarningLevel = 2
	// InfoLevel logging.
	InfoLevel = 3
	// DebugLevel logging.
	DebugLevel = 4
	// DumpLevel logging.
	DumpLevel = 5
)

const (
	// ErrorLabel is a label for a Error message.
	ErrorLabel = "✖"
	// DebugLabel is a label for a debug message.
	DebugLabel = "▶"
	// DumpLabel is a label for a dump message.
	DumpLabel = "▼"
	// InfoLabel is a label for an informative message.
	InfoLabel = "ℹ"
	// SuccessLabel is a label for a success message.
	SuccessLabel = "✔"
	// WarningLabel is a label for a warning message.
	WarningLabel = "!"
)

var (
	// Level to log at. Defaults to info level.
	Level = InfoLevel
	// Color should be enabled for logs.
	Color = true
	// TestMode is enabled.
	TestMode = false
	// Timestamps should be printed.
	Timestamps = false
)

// Fatalf message logs formatted Error then exits with code 1.
func Fatalf(format string, a ...interface{}) {
	if Level >= ErrorLevel {
		a, w := extractLoggerArgs(format, a...)
		l := ErrorLabel
		if Color {
			w = color.Output
			l = color.RedString(l)
		}
		s := fmt.Sprintf(label(format, l), a...)
		fmt.Fprintf(w, s)
		os.Exit(1)
	}
}

// Fataly prints the YAML represtation of an object at Error level then exits with code 1.
func Fataly(name string, obj interface{}) {
	yam, err := SPrintYAML(obj)
	if err != nil {
		Error(err)
		Fatal(obj)
		return
	}
	Fatalf("%s >> \n\n%s\n", name, yam)
}

// Fatal logs Error message then exits with code 1.
func Fatal(a ...interface{}) {
	Fatalf(buildFormat(a...), a...)
}

// Errorf is a formatted Error message.
func Errorf(format string, a ...interface{}) {
	if Level >= ErrorLevel {
		a, w := extractLoggerArgs(format, a...)
		l := ErrorLabel
		if Color {
			w = color.Output
			l = color.RedString(l)
		}
		s := fmt.Sprintf(label(format, l), a...)
		fmt.Fprintf(w, s)
	}
}

// Errory prints the YAML represtation of an object at Error level.
func Errory(name string, obj interface{}) {
	yam, err := SPrintYAML(obj)
	if err != nil {
		Error(err)
		Error(obj)
		return
	}
	Errorf("%s >> \n\n%s\n", name, yam)
}

// Error message.
func Error(a ...interface{}) {
	Errorf(buildFormat(a...), a...)
}

// Infof is a formatted Info message.
func Infof(format string, a ...interface{}) {
	if Level >= InfoLevel {
		a, w := extractLoggerArgs(format, a...)
		l := InfoLabel
		if Color {
			w = color.Output
			l = color.CyanString(l)
		}
		s := fmt.Sprintf(label(format, l), a...)
		fmt.Fprintf(w, s)
	}
}

// Infoy prints the YAML represtation of an object at Info level.
func Infoy(name string, obj interface{}) {
	yam, err := SPrintYAML(obj)
	if err != nil {
		Error(err)
		Info(obj)
		return
	}
	Infof("%s >> \n\n%s\n", name, yam)
}

// Info message.
func Info(a ...interface{}) {
	Infof(buildFormat(a...), a...)
}

// Successf is a formatted Success message.
func Successf(format string, a ...interface{}) {
	if Level >= InfoLevel {
		a, w := extractLoggerArgs(format, a...)
		l := SuccessLabel
		if Color {
			w = color.Output
			l = color.HiGreenString(l)
		}
		s := fmt.Sprintf(label(format, l), a...)
		fmt.Fprintf(w, s)
	}
}

// Successy prints the YAML represtation of an object at Success level.
func Successy(name string, obj interface{}) {
	yam, err := SPrintYAML(obj)
	if err != nil {
		Error(err)
		Success(obj)
		return
	}
	Successf("%s >> \n\n%s\n", name, yam)
}

// Success message.
func Success(a ...interface{}) {
	Successf(buildFormat(a...), a...)
}

// Debugf is a formatted Debug message.
func Debugf(format string, a ...interface{}) {
	if Level >= DebugLevel {
		a, w := extractLoggerArgs(format, a...)
		l := DebugLabel
		if Color {
			w = color.Output
			l = color.MagentaString(l)
		}
		s := fmt.Sprintf(label(format, l), a...)
		fmt.Fprintf(w, s)
	}
}

// Debugy prints the YAML represtation of an object at Debug level.
func Debugy(name string, obj interface{}) {
	yam, err := SPrintYAML(obj)
	if err != nil {
		Error(err)
		Debug(obj)
		return
	}
	Debugf("%s >> \n\n%s\n", name, yam)
}

// Debug message.
func Debug(a ...interface{}) {
	Debugf(buildFormat(a...), a...)
}

// Dumpf is a formatted Dump message.
func Dumpf(format string, a ...interface{}) {
	if Level >= DumpLevel {
		a, w := extractLoggerArgs(format, a...)
		l := DumpLabel
		if Color {
			w = color.Output
			l = color.BlueString(l)
		}
		s := fmt.Sprintf(label(format, l), a...)
		fmt.Fprintf(w, s)
	}
}

// Dumpy prints the YAML represtation of an object at Dump level.
func Dumpy(name string, obj interface{}) {
	yam, err := SPrintYAML(obj)
	if err != nil {
		Error(err)
		Dump(obj)
	}
	Dumpf("%s >> \n\n%s\n", name, yam)
}

// Dump message.
func Dump(a ...interface{}) {
	Dumpf(buildFormat(a...), a...)
}

// Warningf is a formatted Warning message.
func Warningf(format string, a ...interface{}) {
	if Level >= WarningLevel {
		a, w := extractLoggerArgs(format, a...)
		l := WarningLabel
		if Color {
			w = color.Output
			l = color.HiYellowString(l)
		}
		s := fmt.Sprintf(label(format, l), a...)
		fmt.Fprintf(w, s)
	}
}

// Warningy prints the YAML represtation of an object at Warning level.
func Warningy(name string, obj interface{}) {
	yam, err := SPrintYAML(obj)
	if err != nil {
		Error(err)
		Warning(obj)
		return
	}
	Warningf("%s >> \n\n%s\n", name, yam)
}

// Warning message.
func Warning(a ...interface{}) {
	Warningf(buildFormat(a...), a...)
}

// SPrintYAML returns a YAML string for an object and has support for proto messages.
func SPrintYAML(a interface{}) (string, error) {
	var out string
	if m, ok := a.(proto.Message); ok {
		marshaller := &jsonpb.Marshaler{}
		var b bytes.Buffer
		err := marshaller.Marshal(&b, m)
		if err != nil {
			return out, err
		}
		yam, err := yamlconv.JSONToYAML(b.Bytes())
		if err != nil {
			return out, err
		}
		out = string(yam)
	} else {
		b, err := yaml.Marshal(a)
		if err != nil {
			return out, err
		}
		out = string(b)
	}
	return out, nil
}

// PrintYAML prints the YAML string of an object and has support for proto messages.
func PrintYAML(a interface{}) error {
	s, err := SPrintYAML(a)
	if err != nil {
		return err
	}
	fmt.Println(s)
	return nil
}

func extractLoggerArgs(format string, a ...interface{}) ([]interface{}, io.Writer) {
	var w io.Writer = os.Stdout

	if n := len(a); n > 0 {
		// extract an io.Writer at the end of a
		if value, ok := a[n-1].(io.Writer); ok {
			w = value
			a = a[0 : n-1]
		}
	}

	return a, w
}

func label(format, label string) string {
	if Timestamps {
		return labelWithTime(format, label)
	}
	return labelWithoutTime(format, label)
}

func labelWithTime(format, label string) string {
	t := time.Now()
	rfct := t.Format(time.RFC3339)
	if !strings.Contains(format, "\n") {
		format = fmt.Sprintf("%s%s", format, "\n")
	}
	return fmt.Sprintf("%s [%s]  %s", rfct, label, format)
}

func labelWithoutTime(format, label string) string {
	if !strings.Contains(format, "\n") {
		format = fmt.Sprintf("%s%s", format, "\n")
	}
	return fmt.Sprintf("%s  %s", label, format)
}

func buildFormat(f ...interface{}) string {
	var fin string
	for _, i := range f {
		if _, ok := i.(error); ok {
			fin += "%s "
		} else if _, ok := i.(string); ok {
			fin += "%s "
		} else {
			fin += "%#v "
		}
	}
	return fin
}
