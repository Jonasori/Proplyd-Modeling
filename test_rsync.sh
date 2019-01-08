
# Sample rsync script, following Sophia's stuff.

rsync -e "ssh -p [some port number i.e. 554322] -avuhW --progress /path/to/local/dir jpowell@iorek.astro.wesleyan.edu/path/to/remote/

# or, to go from server to local:

rsync -e "ssh -p [45625]" -avuh --progress --timeout=300 --partial-dir=.rsync-partial jpowell@iorek..... /path/to/local
