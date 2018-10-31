//Gradient boosting optimizing RMS. gbt_train.cpp: main function of executable gbt_train
//(c) Daria Sorokina

//gbt_train -t _train_set_ -v _validation_set_ -r _attr_file_ 
//[-a _alpha_value_] [-mu _mu_value_] [-n _boosting_iterations_] [-i _init_random_] [-c rms|roc]
// [-sh _shrinkage_ ] [-sub _subsampling_] [-multi _task_variable_name_] [-smu _shared_mu_]  | -version

#include "Tree.h"
#include "functions.h"
#include "TrainInfo.h"
#include "LogStream.h"
#include "ErrLogStream.h"
#include "bt_definitions.h"
#include "bt_functions.h"

#ifndef _WIN32
#include "thread_pool.h"
#endif

#include <algorithm>
#include <errno.h>
#include <numeric>
#include <cmath>

int main(int argc, char* argv[])
{	
	try{
//0. -version mode	
	if((argc > 1) && !string(argv[1]).compare("-version"))
	{
		LogStream telog;
		telog << "\n-----\nbt_train ";
		for(int argNo = 1; argNo < argc; argNo++)
			telog << argv[argNo] << " ";
		telog << "\n\n";

		telog << "TreeExtra version " << VERSION << "\n";
			return 0;
	}

//1. Analyze input parameters
	//convert input parameters to string from char*
	stringv args(argc); 
	for(int argNo = 0; argNo < argc; argNo++)
		args[argNo] = string(argv[argNo]);
	
	//check that the number of arguments is even (flags + value pairs)
	if(argc % 2 == 0)
		throw INPUT_ERR;

#ifndef _WIN32
	int threadN = 6;	//number of threads
#endif

	TrainInfo ti; //model training parameters
	int topAttrN = 0;  //how many top attributes to output and keep in the cut data 
						//(0 = do not do feature selection)
						//(-1 = output all available features)

	//parse and save input parameters
	//indicators of presence of required flags in the input
	bool hasTrain = false;
	bool hasVal = false; 
	bool hasAttr = false; 

	int treeN = 100;
	double shrinkage = 0.01;
	double subsample = -1;

	for(int argNo = 1; argNo < argc; argNo += 2)
	{
		if(!args[argNo].compare("-t"))
		{
			ti.trainFName = args[argNo + 1];
			hasTrain = true;
		}
		else if(!args[argNo].compare("-v"))
		{
			ti.validFName = args[argNo + 1];
			hasVal = true;
		}
		else if(!args[argNo].compare("-r"))
		{
			ti.attrFName = args[argNo + 1];
			hasAttr = true;
		}
		else if(!args[argNo].compare("-a"))
			ti.alpha = atofExt(argv[argNo + 1]);
		else if(!args[argNo].compare("-mu"))
			ti.mu = atofExt(argv[argNo + 1]); 
        else if(!args[argNo].compare("-smu"))
			ti.smu = atofExt(argv[argNo + 1]); 
		else if(!args[argNo].compare("-n"))
			treeN = atoiExt(argv[argNo + 1]);
		else if(!args[argNo].compare("-i"))
			ti.seed = atoiExt(argv[argNo + 1]);
		else if(!args[argNo].compare("-k"))
			topAttrN = atoiExt(argv[argNo + 1]);
		else if(!args[argNo].compare("-sh"))
			shrinkage = atofExt(argv[argNo + 1]);
		else if(!args[argNo].compare("-sub"))
			subsample = atofExt(argv[argNo + 1]);
        else if(!args[argNo].compare("-multi"))
			ti.multi = args[argNo + 1]; 
		else if(!args[argNo].compare("-c"))
		{
			if(!args[argNo + 1].compare("roc"))
				ti.rms = false;
			else if(!args[argNo + 1].compare("rms"))
				ti.rms = true;
			else
				throw INPUT_ERR;
		}
		else if(!args[argNo].compare("-h"))
#ifndef _WIN32 
			threadN = atoiExt(argv[argNo + 1]);
#else
			throw WIN_ERR;
#endif
		else
			throw INPUT_ERR;
	}//end for(int argNo = 1; argNo < argc; argNo += 2) //parse and save input parameters

	if(!(hasTrain && hasVal && hasAttr))
		throw INPUT_ERR;

	if((ti.alpha < 0) || (ti.alpha > 1))
		throw ALPHA_ERR;
	if(ti.mu < 0)
		throw MU_ERR;

//1.a) Set log file
	LogStream telog;
	LogStream::init(true);
	telog << "\n-----\ngbt_train ";
	for(int argNo = 1; argNo < argc; argNo++)
		telog << argv[argNo] << " ";
	telog << "\n\n";

//1.b) Initialize random number generator. 
	srand(ti.seed);

// if multitask, need vector of INDdata class, vector of CTree

//2. Load data  // for task t in 1..T

    
	INDdata data(ti.trainFName.c_str(), ti.validFName.c_str(), ti.testFName.c_str(), 
				 ti.attrFName.c_str(),ti.multi); // ti.multi is the task variable name
	CTree::setData(data);
	CTreeNode::setData(data);

//2.a) Start thread pool
#ifndef _WIN32    // should we get multithread for each tree from each task?
	TThreadPool pool(threadN);
	CTree::setPool(pool);
#endif

//------------------
	int attrN = data.getAttrN();
	int taskN = data.getTaskN();
	intv usedGroup(attrN,0); // for group penalty
	intvv usedIdv; // for indvidual task penalty
	for(int i = 0; i < taskN; i++)
	{
		intv tmp(attrN,0);
		usedIdv.push_back(tmp);
	}

	// int attrIds[attrN];       
	// fill_n(attrIds, attrN, 0); // initialize all attrIds 0:notused 1:used
	if(topAttrN == -1)
		topAttrN = attrN;

	//idpairv attrCounts;	//counts of attribute importance, need modification for multitask
	idpairvv attrCounts;

	bool doFS = (topAttrN != 0);	//whether feature selection is requested
	if(doFS)
	{//initialize attrCounts


		for(int i = 0; i < taskN; i++)
		{
			idpairv tmp;
			//attrCounts.resize(attrN);
			tmp.resize(attrN);
			for(int attrNo = 0; attrNo < attrN; attrNo++)
			{
				//attrCounts[attrNo].first = attrNo;	//number of attribute	
				//attrCounts[attrNo].second = 0;		//counts
				tmp[attrNo].first = attrNo;	//number of attribute	
				tmp[attrNo].second = 0;		//counts
			}
			attrCounts.push_back(tmp);
		}
	}

	fstream frmscurve("boosting_rms.txt", ios_base::out); //bagging curve (rms)
	frmscurve.close();
	fstream froccurve;
	if(!ti.rms)
	{
		froccurve.open("boosting_roc.txt", ios_base::out); //bagging curve (roc) 
		froccurve.close();
	}

	doublev validTar;
	int validN = data.getTargets(validTar, VALID);

	doublev trainTar;
	int trainN = data.getTargets(trainTar, TRAIN);

	// int sampleN;
	// if(subsample == -1)
	// 	sampleN = trainN;
	// else
	// 	sampleN = (int) (trainN * subsample);
	
	doublev validPreds(validN, 0);
	doublev trainPreds(trainN, 0);
	
    // end for task t in 1..T
    
    iivmap task2Train = data.getTask2TrainMap();

	for(int treeNo = 0; treeNo < treeN; treeNo++) //boosting procedure start
	{
		if(treeNo % 10 == 0)
			cout << "\titeration " << treeNo + 1 << " out of " << treeN << endl;
        // for task t in 1..T
        int taskNo=0;
        cout << "flag1" << endl;

        

        for(iivmap::iterator it = task2Train.begin(); it != task2Train.end(); ++it )

        { 
        	cout << "it->first: "<< it->first << endl;
        	cout << "it->second.size() "<< (it->second).size() << endl;




		if(subsample == -1) 
			data.newBag(it->first);  //pass in taskId
		else
			{   int sampleN = (int) ((it->second).size() * subsample);
			data.newSample(sampleN,it->first); //pass in taskId
		}

		CTree tree(ti.alpha,ti.mu,&usedIdv[taskNo],ti.smu,&usedGroup);  
		tree.setRoot(); 
		tree.resetRoot(trainPreds);
		idpairv stub;
		cout << "flag2" << endl;
		tree.grow(doFS, attrCounts[taskNo]);
		cout << "flag3" << endl;

		//update predictions
		double rmse=0;
		intv tmpTrainId = it->second;
		for(intv::iterator it1 = tmpTrainId.begin(); it1 != tmpTrainId.end(); ++it1 )
		{
			trainPreds[*it1] += shrinkage * tree.predict(*it1, TRAIN);
		}
		cout << "flag4" << endl;
		intv tmpValidId = (data.getTask2ValidMap())[it->first];
		for(intv::iterator it2 = tmpValidId.begin(); it2 != tmpValidId.end(); ++it2 )
		{
			validPreds[*it2] += shrinkage * tree.predict(*it2, VALID);
			rmse += ( validPreds[*it2] -  validTar[*it2] ) * ( validPreds[*it2] -  validTar[*it2] ); 
		}
		cout << "flag5" << endl;
		rmse = sqrt( rmse / tmpValidId.size() ); 

		cout << "flag6" << endl;


		//output
		frmscurve.open("boosting_rms.txt", ios_base::out | ios_base::app); 
		//frmscurve << rmse(validPreds, validTar) << endl;
		frmscurve << "iteration: "<< treeNo + 1 << " task: "<< it->first << " rmse: " << rmse <<endl;
		frmscurve.close();
		cout << "flag7" << endl;




		
		if(!ti.rms)
		{
			froccurve.open("boosting_roc.txt", ios_base::out | ios_base::app); 
			//froccurve << roc(validPreds, validTar) << endl;
			doublev tmpPreds;
			doublev tmpTar;
			for(intv::iterator it3 = tmpValidId.begin(); it3 != tmpValidId.end(); ++it3 )
			{
				tmpPreds.push_back(validPreds[*it3]);
				tmpTar.push_back(validTar[*it3]);
			}


			froccurve << "iteration: "<< treeNo + 1 << " task: "<< it->first << " AUC_ROC: " << roc(tmpPreds, tmpTar) <<endl;
			froccurve.close();
		}

		++taskNo;
		cout << "flag8" << endl;
        
        }// end for task t in 1..T
        

	}


        
        // for task t in 1..T
	// int usedAttrN=0;  // number of used features
	// for(int i=0;i<attrN;i++){
	// 	usedAttrN+=attrIds[i];
	

	//output feature selection results
	if(doFS)
	{	
		if(topAttrN > attrN)
			topAttrN = attrN;
		

		int totalUsedAttrN = accumulate(usedGroup.begin(), usedGroup.end(), 0);

		intv usedCommon(attrN,1); // common used feature across all task

		fstream ffeatures("feature_scores.txt", ios_base::out);

		int taskNo=0;

		for(iivmap::iterator it = task2Train.begin(); it != task2Train.end(); ++it )


		{

		sort(attrCounts[taskNo].begin(), attrCounts[taskNo].end(), idGreater);
		int usedAttrN  = accumulate((usedIdv[taskNo]).begin(), (usedIdv[taskNo]).end(), 0);
		ffeatures << "Number of features used for taskId: "<< it->first << " is " << usedAttrN << "\n";
		ffeatures << "Number of sample for taskId: "<< it->first << " is " << (it->second).size() << "\n";
		ffeatures << "Top " << topAttrN << " features\n";
		for(int attrNo = 0; attrNo < topAttrN; attrNo++)
			ffeatures << data.getAttrName(attrCounts[taskNo][attrNo].first) << "\t"
				<< attrCounts[taskNo][attrNo].second / treeN / (it->second).size() << "\n";
		ffeatures << "\n\nColumn numbers (beginning with 1)\n";

		for(int attrNo = 0; attrNo < topAttrN; attrNo++)
		{
			if( (usedCommon[attrNo] == 1) && (usedIdv[taskNo][attrNo] == 0) )
				usedCommon[attrNo] = 0;

		
			ffeatures << data.getColNo(attrCounts[taskNo][attrNo].first) + 1 << " ";
		}

		ffeatures << "\nLabel column number: " << data.getTarColNo() + 1<< "\n"<<"\n"<<"end of taskId:"<< it->first << "\n" << "\n"<< "\n";
		
		taskNo++;

		}

		int usedCommonAttrN = accumulate(usedCommon.begin(), usedCommon.end(), 0);

		ffeatures << " Total Number of active features: "<< totalUsedAttrN << "\n";
		ffeatures << " Total Number of common active features across all tasks: "<< usedCommonAttrN << "\n";


		ffeatures.close();

		// //output new attribute file
		// for(int attrNo = topAttrN; attrNo < attrN; attrNo++)
		// 	data.ignoreAttr(attrCounts[attrNo].first);
		// data.outAttr(ti.attrFName);
	}

	//output predictions
	fstream fpreds;
	fpreds.open("preds.txt", ios_base::out);
	for(int itemNo = 0; itemNo < validN; itemNo++)
		fpreds << validPreds[itemNo] << endl;
	fpreds.close();
    
    // end for task t in 1..T

//------------------

	}catch(TE_ERROR err){
		te_errMsg((TE_ERROR)err);
		return 1;
	}catch(BT_ERROR err){
		ErrLogStream errlog;
		switch(err) 
		{
			case INPUT_ERR:
				errlog << "Usage: gbt_train -t _train_set_ -v _validation_set_ -r _attr_file_" 
					<< "[-a _alpha_value_] [-mu _mu_value_] [-n _boosting_iterations_] [-i _init_random_] [-c rms|roc]"
					<< " [-sh _shrinkage_ ] [-sub _subsampling_] [-multi _task_variable_name_] [-smu _shared_mu_] | -version\n";
				break;
			case ALPHA_ERR:
				errlog << "Error: alpha value is out of [0;1] range.\n";
				break;
			case MU_ERR:
				errlog << "Error: mu value should be non-negative.\n";
				break;
			case WIN_ERR:
				errlog << "Input error: TreeExtra currently does not support multithreading for Windows.\n"; 
				break;
			default:
				throw err;
		}
		return 1;
	}catch(exception &e){
		ErrLogStream errlog;
		string errstr(e.what());
		errlog << "Error: " << errstr << "\n";
		return 1;
	}catch(...){
		string errstr = strerror(errno);
		ErrLogStream errlog;
		errlog << "Error: " << errstr << "\n";
		return 1;
	}
	return 0;
}
