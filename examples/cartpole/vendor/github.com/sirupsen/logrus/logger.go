package logrus

import (
	"context"
	"io"
	"os"
	"sync"
	"sync/atomic"
	"time"
)

type Logger struct {
	// The logs are `io.Copy`'d to this in a mutex. It's common to set this to a
	// file, or leave it default which is `os.Stderr`. You can also set this to
	// something more adventurous, such as logging to Kafka.
	Out io.Writer
	// Hooks for the logger instance. These allow firing events based on logging
	// levels and log entries. For example, to send errors to an error tracking
	// service, log to StatsD or dump the core on fatal errors.
	Hooks LevelHooks
	// All log entries pass through the formatter before logged to Out. The
	// included formatters are `TextFormatter` and `JSONFormatter` for which
	// TextFormatter is the default. In development (when a TTY is attached) it
	// logs with colors, but to a file it wouldn't. You can easily implement your
	// own that implements the `Formatter` interface, see the `README` or included
	// formatters for examples.
	Formatter Formatter

	// Flag for whether to log caller info (off by default)
	ReportCaller bool

	// The logging level the logger should log at. This is typically (and defaults
	// to) `logrus.Info`, which allows Info(), Warn(), Error() and Fatal() to be
	// logged.
	Level Level
	// Used to sync writing to the log. Locking is enabled by Default
	mu MutexWrap
	// Reusable empty entry
	entryPool sync.Pool
	// Function to exit the application, defaults to `os.Exit()`
	ExitFunc exitFunc
}

type exitFunc func(int)

type MutexWrap struct {
	lock     sync.Mutex
	disabled bool
}

func (mw *MutexWrap) Lock() {
	if !mw.disabled {
		mw.lock.Lock()
	}
}

func (mw *MutexWrap) Unlock() {
	if !mw.disabled {
		mw.lock.Unlock()
	}
}

func (mw *MutexWrap) Disable() {
	mw.disabled = true
}

// Creates a new log. Configuration should be set by changing `Formatter`,
// `Out` and `Hooks` directly on the default logger instance. You can also just
// instantiate your own:
//
//    var log = &Logger{
//      Out: os.Stderr,
//      Formatter: new(JSONFormatter),
//      Hooks: make(LevelHooks),
//      Level: logrus.DebugLevel,
//    }
//
// It's recommended to make this a global instance called `log`.
func New() *Logger {
	return &Logger{
		Out:          os.Stderr,
		Formatter:    new(TextFormatter),
		Hooks:        make(LevelHooks),
		Level:        InfoLevel,
		ExitFunc:     os.Exit,
		ReportCaller: false,
	}
}

func (logger *Logger) newEntry() *Entry {
	entry, ok := log.entryPool.Get().(*Entry)
	if ok {
		return entry
	}
	return NewEntry(logger)
}

func (logger *Logger) releaseEntry(entry *Entry) {
	entry.Data = map[string]interface{}{}
	log.entryPool.Put(entry)
}

// Adds a field to the log entry, note that it doesn't log until you call
// Debug, Print, Info, Warn, Error, Fatal or Panic. It only creates a log entry.
// If you want multiple fields, use `WithFields`.
func (logger *Logger) WithField(key string, value interface{}) *Entry {
	entry := log.newEntry()
	defer log.releaseEntry(entry)
	return entry.WithField(key, value)
}

// Adds a struct of fields to the log entry. All it does is call `WithField` for
// each `Field`.
func (logger *Logger) WithFields(fields Fields) *Entry {
	entry := log.newEntry()
	defer log.releaseEntry(entry)
	return entry.WithFields(fields)
}

// Add an error as single field to the log entry.  All it does is call
// `WithError` for the given `error`.
func (logger *Logger) WithError(err error) *Entry {
	entry := log.newEntry()
	defer log.releaseEntry(entry)
	return entry.WithError(err)
}

// Add a context to the log entry.
func (logger *Logger) WithContext(ctx context.Context) *Entry {
	entry := log.newEntry()
	defer log.releaseEntry(entry)
	return entry.WithContext(ctx)
}

// Overrides the time of the log entry.
func (logger *Logger) WithTime(t time.Time) *Entry {
	entry := log.newEntry()
	defer log.releaseEntry(entry)
	return entry.WithTime(t)
}

func (logger *Logger) Logf(level Level, format string, args ...interface{}) {
	if log.IsLevelEnabled(level) {
		entry := log.newEntry()
		entry.Logf(level, format, args...)
		log.releaseEntry(entry)
	}
}

func (logger *Logger) Tracef(format string, args ...interface{}) {
	log.Logf(TraceLevel, format, args...)
}

func (logger *Logger) Debugf(format string, args ...interface{}) {
	log.Logf(DebugLevel, format, args...)
}

func (logger *Logger) Infof(format string, args ...interface{}) {
	log.Logf(InfoLevel, format, args...)
}

func (logger *Logger) Printf(format string, args ...interface{}) {
	entry := log.newEntry()
	entry.Printf(format, args...)
	log.releaseEntry(entry)
}

func (logger *Logger) Warnf(format string, args ...interface{}) {
	log.Logf(WarnLevel, format, args...)
}

func (logger *Logger) Warningf(format string, args ...interface{}) {
	log.Warnf(format, args...)
}

func (logger *Logger) Errorf(format string, args ...interface{}) {
	log.Logf(ErrorLevel, format, args...)
}

func (logger *Logger) Fatalf(format string, args ...interface{}) {
	log.Logf(FatalLevel, format, args...)
	log.Exit(1)
}

func (logger *Logger) Panicf(format string, args ...interface{}) {
	log.Logf(PanicLevel, format, args...)
}

func (logger *Logger) Log(level Level, args ...interface{}) {
	if log.IsLevelEnabled(level) {
		entry := log.newEntry()
		entry.Log(level, args...)
		log.releaseEntry(entry)
	}
}

func (logger *Logger) Trace(args ...interface{}) {
	log.Log(TraceLevel, args...)
}

func (logger *Logger) Debug(args ...interface{}) {
	log.Log(DebugLevel, args...)
}

func (logger *Logger) Info(args ...interface{}) {
	log.Log(InfoLevel, args...)
}

func (logger *Logger) Print(args ...interface{}) {
	entry := log.newEntry()
	entry.Print(args...)
	log.releaseEntry(entry)
}

func (logger *Logger) Warn(args ...interface{}) {
	log.Log(WarnLevel, args...)
}

func (logger *Logger) Warning(args ...interface{}) {
	log.Warn(args...)
}

func (logger *Logger) Error(args ...interface{}) {
	log.Log(ErrorLevel, args...)
}

func (logger *Logger) Fatal(args ...interface{}) {
	log.Log(FatalLevel, args...)
	log.Exit(1)
}

func (logger *Logger) Panic(args ...interface{}) {
	log.Log(PanicLevel, args...)
}

func (logger *Logger) Logln(level Level, args ...interface{}) {
	if log.IsLevelEnabled(level) {
		entry := log.newEntry()
		entry.Logln(level, args...)
		log.releaseEntry(entry)
	}
}

func (logger *Logger) Traceln(args ...interface{}) {
	log.Logln(TraceLevel, args...)
}

func (logger *Logger) Debugln(args ...interface{}) {
	log.Logln(DebugLevel, args...)
}

func (logger *Logger) Infoln(args ...interface{}) {
	log.Logln(InfoLevel, args...)
}

func (logger *Logger) Println(args ...interface{}) {
	entry := log.newEntry()
	entry.Println(args...)
	log.releaseEntry(entry)
}

func (logger *Logger) Warnln(args ...interface{}) {
	log.Logln(WarnLevel, args...)
}

func (logger *Logger) Warningln(args ...interface{}) {
	log.Warnln(args...)
}

func (logger *Logger) Errorln(args ...interface{}) {
	log.Logln(ErrorLevel, args...)
}

func (logger *Logger) Fatalln(args ...interface{}) {
	log.Logln(FatalLevel, args...)
	log.Exit(1)
}

func (logger *Logger) Panicln(args ...interface{}) {
	log.Logln(PanicLevel, args...)
}

func (logger *Logger) Exit(code int) {
	runHandlers()
	if log.ExitFunc == nil {
		log.ExitFunc = os.Exit
	}
	log.ExitFunc(code)
}

//When file is opened with appending mode, it's safe to
//write concurrently to a file (within 4k message on Linux).
//In these cases user can choose to disable the lock.
func (logger *Logger) SetNoLock() {
	log.mu.Disable()
}

func (logger *Logger) level() Level {
	return Level(atomic.LoadUint32((*uint32)(&log.Level)))
}

// SetLevel sets the logger level.
func (logger *Logger) SetLevel(level Level) {
	atomic.StoreUint32((*uint32)(&log.Level), uint32(level))
}

// GetLevel returns the logger level.
func (logger *Logger) GetLevel() Level {
	return log.level()
}

// AddHook adds a hook to the logger hooks.
func (logger *Logger) AddHook(hook Hook) {
	log.mu.Lock()
	defer log.mu.Unlock()
	log.Hooks.Add(hook)
}

// IsLevelEnabled checks if the log level of the logger is greater than the level param
func (logger *Logger) IsLevelEnabled(level Level) bool {
	return log.level() >= level
}

// SetFormatter sets the logger formatter.
func (logger *Logger) SetFormatter(formatter Formatter) {
	log.mu.Lock()
	defer log.mu.Unlock()
	log.Formatter = formatter
}

// SetOutput sets the logger output.
func (logger *Logger) SetOutput(output io.Writer) {
	log.mu.Lock()
	defer log.mu.Unlock()
	log.Out = output
}

func (logger *Logger) SetReportCaller(reportCaller bool) {
	log.mu.Lock()
	defer log.mu.Unlock()
	log.ReportCaller = reportCaller
}

// ReplaceHooks replaces the logger hooks and returns the old ones
func (logger *Logger) ReplaceHooks(hooks LevelHooks) LevelHooks {
	log.mu.Lock()
	oldHooks := log.Hooks
	log.Hooks = hooks
	log.mu.Unlock()
	return oldHooks
}
