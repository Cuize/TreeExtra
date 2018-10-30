// Bagged Trees / Tree.h: interface of class CTree
//
// (c) Daria Sorokina

#pragma once
#include "TreeNode.h"

#ifndef _WIN32
#include "thread_pool.h"
#endif

//Regression Tree model
class CTree
{
public:
	//set function for static data pointer
    void setData(INDdata& data){pData = &data;}

#ifndef _WIN32
	void setPool(TThreadPool& pool){pPool = &pool;}
#endif

	//constructor
	CTree(double alpha = 0 ,double mu = 0 ,intv* pUsedIdv = NULL, double smu = 0, intv* pUsedGroup = NULL); 

	//grows a tree, increases attribute counts
	void grow(bool doFS, idpairv& attrCounts);

	//saves the tree into the binary file
	void save(const char* fileName);

	//loads the tree from the binary file
	void load(fstream& fload);

	//calculates prediction of the model for a single item
	double predict(int itemNo, DATA_SET dset);

	//loads data into the root
	void setRoot();

	//input: predictions for train set data points produced by the rest of the model (not by this tree)	
	//Changes ground truth to residuals in the root train set
	void resetRoot(doublev& othpreds);

private:
    INDdata* pData;	//data access pointer

#ifndef _WIN32
    TThreadPool* pPool;	//thread pool pointer
	TCondition nodesCond;	//condition, used for multithreading control 
#endif

	CTreeNode root;		//root of the tree
	double alpha;		//training parameter: controls size of the tree
	double mu;          //training parameter: penalty on new split variable (idv task)
	double smu;         //training parameter: penalty on new split variable (across all tasks)
	intv* pUsedIdv;     
	intv* pUsedGroup;
};

#ifndef _WIN32 
//Information required for a single node splitting job to run. Used for multithreading
struct JobData
{	
	JobData(nodeip in_curNH, nodehstack* in_pNodes, TCondition* in_pNodesCond, int* in_pToDoN, 
			idpairv* in_pAttrCounts, double in_b, double in_H, double muIn, intv* pUsedIdvIn,double smuIn, intv* pUsedGroupIn):
	curNH(in_curNH), pNodes(in_pNodes), pNodesCond(in_pNodesCond), pToDoN(in_pToDoN), 
	pAttrCounts(in_pAttrCounts), b(in_b), H(in_H), mu(muIn), pUsedIdv(pUsedIdvIn),smu(smuIn), pUsedGroup(pUsedGroupIn) {}

	nodeip curNH; 
	nodehstack* pNodes;
	TCondition* pNodesCond;
	int* pToDoN;
	double alpha;
	double mu;
	double smu;
	intv* pUsedIdv;
	intv* pUsedGroup;
	idpairv* pAttrCounts;
	double b;
	double H;
};
#endif
