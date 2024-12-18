export LOG_LEVEL="INFO"

# setup glog
export GLOG_alsologtostderr=1
# export GLOG_log_dir=./logs
if [ "$LOG_LEVEL" = "ERROR" ]; then
    export GLOG_v=0
elif [ "$LOG_LEVEL" = "WARNING" ]; then
    export GLOG_v=1
elif [ "$LOG_LEVEL" = "INFO" ]; then
    export GLOG_v=2
elif [ "$LOG_LEVEL" = "DEBUG" ]; then
    export GLOG_v=3
else
    echo "Invalid log level"
    exit 1
fi