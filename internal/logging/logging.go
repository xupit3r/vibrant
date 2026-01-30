package logging

import (
"io"
"os"
"path/filepath"

"github.com/sirupsen/logrus"
)

var log *logrus.Logger

// Init initializes the logger with the given configuration
func Init(level, logFile string, console bool) error {
log = logrus.New()

// Set log level
lvl, err := logrus.ParseLevel(level)
if err != nil {
lvl = logrus.InfoLevel
}
log.SetLevel(lvl)

// Set formatter
log.SetFormatter(&logrus.TextFormatter{
FullTimestamp:   true,
TimestampFormat: "2006-01-02 15:04:05",
})

// Set output
var writers []io.Writer

if console {
writers = append(writers, os.Stderr)
}

if logFile != "" {
// Ensure directory exists
dir := filepath.Dir(logFile)
if err := os.MkdirAll(dir, 0755); err != nil {
return err
}

file, err := os.OpenFile(logFile, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
if err != nil {
return err
}
writers = append(writers, file)
}

if len(writers) > 0 {
log.SetOutput(io.MultiWriter(writers...))
}

return nil
}

// Get returns the logger instance
func Get() *logrus.Logger {
if log == nil {
log = logrus.New()
}
return log
}

// Convenience functions
func Debug(args ...interface{}) {
Get().Debug(args...)
}

func Debugf(format string, args ...interface{}) {
Get().Debugf(format, args...)
}

func Info(args ...interface{}) {
Get().Info(args...)
}

func Infof(format string, args ...interface{}) {
Get().Infof(format, args...)
}

func Warn(args ...interface{}) {
Get().Warn(args...)
}

func Warnf(format string, args ...interface{}) {
Get().Warnf(format, args...)
}

func Error(args ...interface{}) {
Get().Error(args...)
}

func Errorf(format string, args ...interface{}) {
Get().Errorf(format, args...)
}

func Fatal(args ...interface{}) {
Get().Fatal(args...)
}

func Fatalf(format string, args ...interface{}) {
Get().Fatalf(format, args...)
}
