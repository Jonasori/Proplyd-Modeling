## PREAMBLE
I have been working to characterize two disks around stars in a young binary in the Orion Nebula cluster, using this code to do so. There are loads more files involved than just this one, but as it will take me a while to reorganize my system to be well-suited to life on git, this junk will have to do for now.




## INTENTION
Given a range of parameters and a the appropriate molecular line data, create a best-fit model of the V2434 Ori system.
	Takes:		Model disk parameter ranges,
				Data file (for chiSq())
	Creates:	Best-fit model system (as fits, im, vis, uvf files)
				Long and short log files (complete and summarized grid-search data)

	Use tools.py for channel map image, residuals (not yet implemented), and other helper functions.
	THINGS TO REMEMBER:
	1. All file extensions on file names are added in the functions, so don't put them in the variable name.




## TO DO

	- Figure out how Cail's damn channel map plotter works.






## STRUCTURE

EXPLANATION OF FUNCTIONS

	makeModel(diskParams, outputName, DI)
		- diskParams: list of vals
		- outputName: string
		- DI: Disk Index (0 or 1), indicates which disk is being fit for (A = 0)

	sumDisks(fileNameA, fileNameB, outputName)
		- fileNameA/B: string
		- outputName: string

	chiSq(infile)
		- infile: filename of model to compare to data

	gridSearch(variedDiskParams, staticDiskParams, DI)
		- variedDiskParams: list of vals
		- staticDiskParams: list of floats
		- DI: Disk Index (0 or 1), indicates which disk is being fit for (A = 0)

	fullRun(diskParams, outputName, DI, fileNameA, fileNameB, outputName, variedDiskParams, outNameVarieds, outNameStatic, modelDisk, dataDisk)
		- all these as above
