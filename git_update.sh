#
#
#
DATE=`date`
WHO=`whoami`

git add . && git commit -m "$WHO $DATE" && git push

