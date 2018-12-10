// TreeNode.cpp: implementation of the CTreeNode class.
//
// Implementation notes:
//
// 1. The tree is binary: each node has either two offsprings or none
// 2. The tree can use continuous and boolean attributes (no nominals).
// 3. Cases with missing values go to both branches with coefficients proportional to 
// the distribution of other cases. 
//		3.a) Special split: all cases with missing values go to one brance, the rest go other way
// 4. We keeps several indexes of the data in the node, each copy sorted by values of one of the 
// attributes. Because of this, splitting of each node (except for the root) takes linear time. 
// 5. Some stl variables are implemented as pointers in order to ensure that unused memory can be 
// freed fast enough. ( someData.clear() does not free memory, delete pSomeData does ) 
// 
// (c) Daria Sorokina

#include "TreeNode.h"
#include "functions.h"

#include <fstream>
#include <math.h>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cmath>


INDdata* CTreeNode::pData;

//Constructor. If the node is a root, download info about the train set.
CTreeNode::CTreeNode(): 
	left(0), right(0), pAttrs(NULL), pSorted(NULL), pItemSet(NULL),variance(0)
{
	
}

//Destructor. Deletes subtrees and node specific contents
CTreeNode::~CTreeNode()
{
	del();
	if(pItemSet)
		delete pItemSet;
	if(pAttrs)
		delete pAttrs;
	if(pSorted)
		delete pSorted;
}

//Deletes a subtree with a root in this node. It is recursive because it calls destructor.
void CTreeNode::del()
{
	if(left)
	{
		delete left;
		left = NULL;
	}
	if(right)
	{
		delete right;
		right = NULL;
	}
}

//assignment operator
//Kills the subtree and all information attached to the original node.
//Copies all internal node information except for the subtree (copies values of pointers, but not the subtree itself)
//Important side effect to remember: both nodes point to the same subtree after the assigniment.
CTreeNode& CTreeNode::operator=(const CTreeNode& rhs)
{
	//kill the subtree of the original node
	del();

	//copy node specific contents of the copied node to the original node (not only pointers)
	if(pItemSet)
		delete pItemSet;
	if(rhs.pItemSet)
		pItemSet = new ItemInfov(*rhs.pItemSet);
	else
		pItemSet = NULL;
	
	if(pAttrs)
		delete pAttrs;
	if(rhs.pAttrs)
		pAttrs = new intv(*rhs.pAttrs);
	else
		pAttrs = NULL;
	
	if(pSorted)
		delete pSorted;
	if(rhs.pSorted)
		pSorted = new dipairvv(*rhs.pSorted);
	else
		pSorted = NULL;

	//copy pointers to subtrees and dataset class
	left = rhs.left;		
	right = rhs.right;		

	//copy nonpointer contents
	splitting = rhs.splitting;

	return *this;
}

//copy constructor
//Similar to the assignment operator except that it does not have to erase old content
CTreeNode::CTreeNode(const CTreeNode& rhs)
{	
	//copy node specific contents of the copied node to the original node (not only pointers)
	if(rhs.pItemSet)
		pItemSet = new ItemInfov(*rhs.pItemSet);
	else
		pItemSet = NULL;
	
	if(rhs.pAttrs)
		pAttrs = new intv(*rhs.pAttrs);
	else
		pAttrs = NULL;
	
	if(rhs.pSorted)
		pSorted = new dipairvv(*rhs.pSorted);
	else
		pSorted = NULL;

	//copy pointers to subtrees and dataset class
	left = rhs.left;		
	right = rhs.right;		

	//copy nonpointer contents
	splitting = rhs.splitting;
}

//Deletes old tree, gets data from the dataset container into the node 
//This function is intended for root nodes only
void CTreeNode::setRoot()
{


	del();	//delete old tree

	variance=0;

	if(pAttrs == NULL)
		pAttrs = new intv();
	pData->getActiveAttrs(*pAttrs);

	if(pSorted == NULL)
		pSorted = new dipairvv();
	pData->getSortedData(*pSorted);
	
	if(pItemSet == NULL)
		pItemSet = new ItemInfov();
	pData->getCurBag(*pItemSet);

	// if(pPrefixedSum == NULL)
	// 	pPrefixedSum = new floatvv();
	// pData->getPrefixedSum(*pPrefixedSum,attrIds);





}

//input: predictions for train set data points produced by the rest of the model (not by this tree)	
//Changes ground truth to residuals in the root train set
void CTreeNode::resetRoot(doublev& othpreds)
{	double sums=0;
	double square=0;
	int count=0;
	for(ItemInfov::iterator itemIt = pItemSet->begin();	itemIt != pItemSet->end();	itemIt++){
		itemIt->response -= othpreds[itemIt->key];
		count += 1; 
		sums += itemIt->response;
		square += (itemIt->response)*(itemIt->response);
	}
	variance = square - sums*sums/count;
}

//This function is used for prediction. It passes the case from the parent to child node(s).
//The node must not be a leaf.
//input:
	//itemNo - case id
	//inCoef - coefficient the case came into the node with
	//dset - data set type
//out:
	//lOutCoef - coefficient the case goes to the left child node with 
	//rOutCoef - coefficient the case goes to the right child node with
void CTreeNode::traverse(int itemNo, double inCoef, double& lOutCoef, double& rOutCoef, DATA_SET dset)
{
	double value = pData->getValue(itemNo, splitting.divAttr, dset);  //value of the split attribute
	double lCoef = splitting.leftCoef(value); //left coefficient for this value 
	double rCoef = 1 - lCoef; //right coefficient for this value
	lOutCoef = lCoef * inCoef;
	rOutCoef = rCoef * inCoef;
}


// Grows 2 child nodes of this node, using RMSE as split quality criterion
// Returns true if succeeds, false if this node becomes a leaf
// input: alpha - min possible ratio of internal node train subset volume to the whole train set size, 
//		when surpassed,	the node becomes a leaf
bool CTreeNode::split(double alpha, double rootVar, double* pEntropy, double mu, int *attrIds, int s, int* numUsed, double* compN)
{	
//1. check basic leaf conditions
	double nodeV, nodeSum, squares, realNodeV;
	bool StD0 = getStats(nodeV, nodeSum, squares, realNodeV);

	if((realNodeV / pData->getTrainV() < alpha) || StD0)
	{
		makeLeaf(nodeSum / nodeV);
		return false;
	}

//2. Find the best split
//evaluate all good splits and choose the one with the best evaluation, also approximate split by groupTest and binarySearch
	// conditions to use setGroupSplit..
	int remain = max(1,s - (*numUsed));
	int n = (int)pItemSet->size();
	int d = pData->getAttrN();
	int d0 = pAttrs->size();
	double comp1 = d0;
	double comp2 =( 3 + max(1.0,log(n))) * log2(d);
	bool trigger = (( remain == 1 ) && (  comp2 < comp1 ));

	// cout<< "group: " << group << endl;
	// cout<< "d: " << d << endl;
	// cout << "n: "<< n << endl;
	// cout << "triggered :" << trigger << endl;

	bool notFound = ( pData->useCoef() ? setSplitMV(nodeV, nodeSum, squares, rootVar, mu, attrIds) : ( (!trigger) ? setSplit(nodeV, nodeSum, squares, rootVar, mu, attrIds, numUsed, compN) : setGroupSplit(nodeV, nodeSum, squares, rootVar, mu, attrIds, s, numUsed, compN) ) );	//finds and sets best split

	if(notFound)
	{//no splittings or they disappeared because of tiny coefficients. This node becomes a leaf
		makeLeaf(nodeSum / nodeV);
		return false;
	}

	//at this point the best splitting is found and set
//2a. If requested, return entropy of the winning feature
	if(pEntropy != NULL)
		*pEntropy = getEntropy(splitting.divAttr);

//3. Generate two child nodes
	
	left = new CTreeNode();
	right = new CTreeNode();

	int itemN = (int)pItemSet->size();

	left->pItemSet = new ItemInfov();
	right->pItemSet = new ItemInfov();
	left->pItemSet->reserve(itemN);
	right->pItemSet->reserve(itemN);

	intv leftHash, rightHash; //correspondence between current items and new items
	//initialize with -1 (no new item corresponds to this one)
	leftHash.resize(itemN, -1);
	rightHash.resize(itemN, -1);

	//allocate cases from the training subset of the parent node in child nodes following the chosen split 
	for(int itemNo = 0; itemNo < itemN; itemNo++)
	{
		//get value of the attribute divAttr for the current training case
		ItemInfo& curItem = (*pItemSet)[itemNo];
		double value = pData->getValue(curItem.key, splitting.divAttr, TRAIN); 
		
		//calculate coefficients of current training case
		double lCoef = splitting.leftCoef(value);
		double rCoef = 1 - lCoef; 
		double curCoef = curItem.coef;
		double newLCoef = lCoef * curCoef;
		double newRCoef = rCoef * curCoef;
		
		if(newLCoef)
		{//put case #itemNo in left branch
			left->pItemSet->push_back(curItem);
			left->pItemSet->back().coef = newLCoef;
			leftHash[itemNo] = (int)left->pItemSet->size() - 1;
		}
		
		if(newRCoef)
		{//put case #itemNo in right branch
			right->pItemSet->push_back(curItem);
			right->pItemSet->back().coef = newRCoef;
			rightHash[itemNo] = (int)right->pItemSet->size() - 1;
		}
	}//end for(int itemNo = 0; itemNo < itemN; itemNo++)

	//create sorted vectors in child nodes
	// if(!trigger)
	// {

		int defAttrN = (int)pAttrs->size();
		left->pSorted = new dipairvv(defAttrN);
		right->pSorted = new dipairvv(defAttrN);
		
		for(int attrNo = 0; attrNo < defAttrN; attrNo++)
		{
			//reserve space 
			(*left->pSorted)[attrNo].reserve((int)left->pItemSet->size());
			(*right->pSorted)[attrNo].reserve((int)right->pItemSet->size());
			
			//insert pairs in childrens sorted vectors in the same order, update item # through hash
			for(dipairv::iterator pvIt = (*pSorted)[attrNo].begin();
				pvIt!=(*pSorted)[attrNo].end(); pvIt++)
			{
				int leftNo = leftHash[pvIt->second];
				int rightNo = rightHash[pvIt->second];
				if(leftNo != -1)
					(*left->pSorted)[attrNo].push_back(dipair(pvIt->first, leftNo));
				if(rightNo != -1)
					(*right->pSorted)[attrNo].push_back(dipair(pvIt->first, rightNo));
			}
		}
	// }
	//clean the parent node
	delete pItemSet;
	pItemSet = NULL;

	delete pSorted;
	pSorted = NULL;

	//move/init attribute set
	left->pAttrs = pAttrs;
	right->pAttrs = new intv(*pAttrs);
	pAttrs = NULL;

	return true;
// }
}


//returns several summaries of the prediction values set in this node
// output:
	//nodeV - sum of squared coefficients
	//nodeSum - sum of predictions respectively multiplied by squared coefficients
	//realNodeV - sum of coefficients
//returns false if there are different response values, true - if all values are the same
bool CTreeNode::getStats(double& nodeV, double& nodeSum, double& squares, double& realNodeV)
{
	nodeSum = 0;
	nodeV = 0;
	squares = 0;
	realNodeV = 0;

	double firstResp = (*pItemSet)[0].response;	//response of the first item
	bool StD0 = true;	//whether all response values are the same (StD == 0)

	//calculate nodeV, nodeSum and realNodeV
	if(pData->useCoef())
		for(ItemInfov::iterator itemIt = pItemSet->begin();	itemIt != pItemSet->end();	itemIt++)
		{// update every sum with the info from this item
			double& coef = itemIt->coef;
			double coef_sq = coef * coef;
			nodeV += coef_sq;
			nodeSum += coef_sq * itemIt->response;
			realNodeV += coef;

			if(itemIt->response != firstResp)
				StD0 = false;
		}
	else
	{//faster calculations for the data without missing values
		for(ItemInfov::iterator itemIt = pItemSet->begin();	itemIt != pItemSet->end();	itemIt++)
		{
			nodeV++;
			nodeSum += itemIt->response;
			squares += (itemIt->response)*(itemIt->response);
			if(itemIt->response != firstResp)
				StD0 = false;
		}
		realNodeV = nodeV;
	}

	return StD0;
}

//returns sum of coefficients, old definition of node volume
double CTreeNode::getNodeV()
{
	if(pData->useCoef())
	{
		double realNodeV = 0;
		for(ItemInfov::iterator itemIt = pItemSet->begin();	itemIt != pItemSet->end();	itemIt++)
			realNodeV += itemIt->coef;
		return realNodeV;
	}
	else
	//faster calculations for the data without missing values
		return pItemSet->size();
}

double CTreeNode::getEntropy(int attrId)
{//get entropy of this feature in this node
	
	//aggregate soft counts for each value of attrId feature in the node's portion of the training data
	ddmap valCounts; //maps values to counts
	double mvCount = 0; //a separate counter for missing values, as map does not take QNAN correctly
	for (ItemInfov::iterator itemIt = pItemSet->begin(); itemIt != pItemSet->end(); itemIt++)
	{
		double value = pData->getValue(itemIt->key, attrId, TRAIN);
		if (isnan(value))
			mvCount += itemIt->coef;
		else
			valCounts[value] += itemIt->coef;
	}
	
	//calculate entropy
	double nodeV = getNodeV();
	double entropy = 0;
	for(ddmap::iterator vcIt = valCounts.begin(); vcIt != valCounts.end(); vcIt++)
		entropy -= vcIt->second / nodeV * log(vcIt->second / nodeV) / log(2.0);
	if (mvCount > 0)
		entropy -= mvCount / nodeV * log(mvCount / nodeV) / log(2.0);
	return entropy;
} 
	
//This node becomes a leaf
//Cleans out node contents that are not used anymore and saves prediction of the node
void CTreeNode::makeLeaf(double nodeMean)
{
	if(pAttrs)
		delete pAttrs;
	pAttrs = NULL;

	if(pSorted)
		delete pSorted;
	pSorted = NULL;

	if(pItemSet)
		delete pItemSet;
	
	pItemSet = new ItemInfov(1);
	(*pItemSet)[0].response = nodeMean;
}

//Chooses and sets best mse split over all attributes and values when no missing values are
// present.
//To compare mse values of splits, we need to calculate only 2 of 3 squared sum components.
//For each attribute it estimates all splits in _one_pass_ over the sorted data, i.e. O(N)
//	(trivial algorithm would produce much less code, but the running time would be O(N^2) )
//Side effect: removes exhausted attributes from the node attribute set
//in:
	// nodeV - size (volume) of the training subset
	// nodeSum - sum of response values in the training subset
//out: true, if best split found. false, if there were no splits

bool CTreeNode::setSplit(double nodeV, double nodeSum, double squares, double rootVar, double mu, int *attrIds, int *numUsed, double *compN)
{
	double bestEval = QNAN; //current value for the best evaluation
	SplitInfov bestSplits; // all splits that have best (identical) evaluation

	int numBool = 0;
	

	

	for(int attrNo = 0; attrNo < (int)pAttrs->size();)
	{
		int attr = (*pAttrs)[attrNo];
		double penalty = (1 - attrIds[attr])*mu; // 0<=mu<1 penalty
		if(pData->boolAttr(attr))	
		{//boolean attribute
			//there is exactly one split for a boolean attribute, evaluate it
			numBool += 1;

			SplitInfo boolSplit(attr, 0.5);
			double eval = evalBool(boolSplit, nodeV, nodeSum, squares, rootVar) + penalty;
			if(isnan(eval))
			{//boolean attribute is not valid anymore, remove it
				pAttrs->erase(pAttrs->begin() + attrNo);	
				pSorted->erase(pSorted->begin() + attrNo);
			}
			else 
			{//save if this is one of the best splits
				if(isnan(bestEval) || (eval < bestEval))
				{
					bestEval = eval;
					bestSplits.clear();
				}
				if(eval == bestEval)
					bestSplits.push_back(SplitInfo(boolSplit));

				attrNo++;
			}
		}
		else //continuous attribute 
		{//candidate splits are installed between all pairs of neighbour values (values are sorted)
		 //all splits are calculated in one pass

			bool newSplits = false;	//true if any splits were added for this attribute
		
			//traverse pSorted[attrNo], create crisp splits between pairs of cases w diff response
				//and evaluate on the fly
			
			//parameters that will be changing for different splits
			double volume1 = 0;						
			double volume2 = nodeV;
			double sum1 = 0;						
			double sum2 = nodeSum;

			//parameters of traverse
			bool prevDiff, curDiff; //whether prev or cur attrval had diff response values
			double prevAttrVal; //previous value of the attribute
			double prevResp = QNAN; //response value if same for prev attrval 
			double curAttrVal; //current value of attribute
			double curResp; //current response, if the same

			//parameters of the set of points that moves from one node to the other
			double prevTraV, prevTraSum; //for the previous block
			double curTraV, curTraSum; //for the current block
			prevTraV = 0; prevTraSum = 0;

			dipairv* pSortedVals = &(*pSorted)[attrNo];
			dipairv::iterator pairIt = pSortedVals->begin();
			while(pairIt != pSortedVals->end())
			{//on each iteration of this cycle collect info about the block of cases with the
				//same value of the attribute and if needed, evaluate the split right before it.
				
				//initialize current traverse parameters
				curAttrVal = pairIt->first;
				curResp = (*pItemSet)[pairIt->second].response;
				curDiff = false;
				curTraV = 0;	
				curTraSum = 0;	

				//get next block, update transition parameters
				dipairv::iterator sortedEnd = pSortedVals->end();
				for(;(pairIt != sortedEnd) && (pairIt->first == curAttrVal); pairIt++)
				{
					ItemInfo& item = (*pItemSet)[pairIt->second];
					curTraV ++;
					curTraSum += item.response;
					if(!curDiff && (item.response != curResp))
						curDiff = true;
				}

				//if there are different responses in previous and current block 
					//build and evaluate the split between them
				if(!isnan(prevResp) && (prevDiff || curDiff || (prevResp != curResp)))
				{
					newSplits = true;
					//calculate the "short mse" of the new split - parts of sum of se that are different for different splits
					//1. update those parameters that we keep
					volume1 += prevTraV;
					volume2 -= prevTraV;
					sum1 += prevTraSum;
					sum2 -= prevTraSum;

					//2. calculate short mse of the split from them
					double leftRatio = volume1 / nodeV;

					double mean1 = sum1 / volume1;
					double mean2 = sum2 / volume2;

					double sqErr1 =  - mean1 * sum1;
					double sqErr2 =  - mean2 * sum2;

					double eval = ( sqErr1 + sqErr2 + squares )/rootVar + penalty;
			
					//evaluate the split point, if it is the best (one of the best) so far, keep it
					if(isnan(bestEval) || (eval < bestEval))
					{
						bestEval = eval;
						bestSplits.clear();
					}
					if(eval == bestEval)
					{
						//create actual split with the split point halfway between attr values
						SplitInfo goodSplit(attr, (curAttrVal + prevAttrVal) / 2, leftRatio);
						bestSplits.push_back(goodSplit);
					}

					//"restart" prev parameters with this block
					prevTraV = curTraV;
					prevTraSum = curTraSum;
				}//end if(!isnan(prevResp) && (prevDiff || curDiff || (prevResp != curResp)))
				else
				{//block was not used, increas "prev" parameters
					prevTraV += curTraV;
					prevTraSum += curTraSum;
				}

				//update previous traverse parameters with values of current
				prevDiff = curDiff;
				prevResp = curResp;
				prevAttrVal = curAttrVal;
			}//end while(pairIt != pSortedVals->end())				

			//if an attribute is exhausted, delete it, shift to next iteration
			if(!newSplits)
			{
				pAttrs->erase(pAttrs->begin() + attrNo);
				pSorted->erase(pSorted->begin() + attrNo);
			}
			else
				attrNo++;
		}//end		if(pData->boolAttr(attr)) else //continuous attribute
	}//end 	for(int attrNo = 0; attrNo < (int)pAttrs->size();)
	// cout<<"# of active features: "<<pAttrs->size()<<endl;
	// cout<<"# of active bool features: "<<numBool<<endl;
	// cout<<"# of sample: "<<nodeV<<endl;
	// cout<<"approximate computation: "<< pAttrs->size() * pItemSet->size()<<endl;
	*compN += (double)(pAttrs->size() * pItemSet->size())/(double)pData->getTrainN(); 
	


	//choose a random split from those with best mse
	if(!isnan(bestEval))
	{
		int bestSplitN = (int)bestSplits.size();
		int randSplit = rand() % bestSplitN;
		splitting = bestSplits[randSplit];
		if(pData->boolAttr(splitting.divAttr))
		{//one can split only once on boolean attribute, remove it from the set of attributes
			int attrNo = erasev(pAttrs, splitting.divAttr);
			pSorted->erase(pSorted->begin() + attrNo);	
				//it is an empty vector (the attribute is boolean), but we still need to remove it
		}
		if(attrIds[splitting.divAttr] == 0)
			*numUsed += 1;

		attrIds[splitting.divAttr] = 1;
	}
	return isnan(bestEval);
}

//Chooses the best split when missing values are present. Same algorithm as in setSplit with the following additions:
//a) Cases with missing values go to both branches with coefficients proportional to the distribution of other cases. 
//b) Cases are weighted by their squared coefficients when a leaf value is calculated
//c) Leaf values are weighted by linear coefficients when a prediction for a single data point is calculated
//d) There is a special split evaluated for each attribute with missing values: cases without 
//	missing values go left, cases with missing values go right
//in:
	//nodeV - sum of squared coefficients
	//nodeSum - sum of predictions respectively multiplied by squared coefficients
//out: true, if best split found. false, if there were no splits
bool CTreeNode::setSplitMV(double nodeV, double nodeSum, double squares, double rootVar, double mu, int *attrIds)
{
	double bestEval = QNAN; //current value for the best evaluation
	SplitInfov bestSplits; // all splits that have best (identical) evaluation

	for(int attrNo = 0; attrNo < (int)pAttrs->size();)
	{
		int attr = (*pAttrs)[attrNo];
		double penalty = (1 - attrIds[attr]) * mu; // scaled penalty
		bool newSplits = false;	//true if any splits were added for this attribute

		//collect info about missing values
		double missSum = 0; //sum of responses of mv cases (multiplied by sq coefs)
		double missV = 0; //volume of mv in the node (sum of sq coefs)
		for(ItemInfov::iterator itemIt = pItemSet->begin(); itemIt != pItemSet->end(); itemIt++)
			if(isnan(pData->getValue(itemIt->key, attr, TRAIN)))
			{
				double coef_sq = itemIt->coef * itemIt->coef;
				missSum += coef_sq * itemIt->response;
				missV += coef_sq;
			}

		if(missV && (missV != nodeV))
		{//evaluate a special split: missing vs not missing
			newSplits = true;

			double nmSum = nodeSum - missSum;
			double nmV = nodeV - missV;
			double mean1 = nmSum / nmV;
			double mean2 = missSum / missV;
			double sqErr1 =  - 2 * mean1 * nmSum + nmV * mean1 * mean1;
			double sqErr2 = - 2 * mean2 * missSum + missV * mean2 * mean2;
			double eval = ( sqErr1 + sqErr2 + squares )/rootVar + penalty;
			
			//if it is the best (one of the best) so far, keep it
			if(isnan(bestEval) || (eval < bestEval))
			{
				bestEval = eval;
				bestSplits.clear();
			}
			if(eval == bestEval)
			{//create actual split with QNAN indicating that this is a special non-missing vs missing split
				SplitInfo goodSplit(attr, QNAN, 0);
				bestSplits.push_back(goodSplit);
			}				
		}

		if(pData->boolAttr(attr))	
		{//boolean attribute
			//there is only one non-special split for a boolean attribute, evaluate it
			SplitInfo boolSplit(attr, 0.5);
			double eval = evalBoolMV(boolSplit, nodeV, nodeSum, squares, rootVar, missV, missSum) + penalty;
			if(!isnan(eval))
			{//save if this is one of the best splits
				newSplits = true;
				if(isnan(bestEval) || (eval < bestEval))
				{
					bestEval = eval;
					bestSplits.clear();
				}
				if(eval == bestEval)
					bestSplits.push_back(SplitInfo(boolSplit));
			}
		}
		else //continuous attribute 
		{//candidate splits are installed between all pairs of neighbour values (values are sorted)
		 //all splits are calculated in one pass

			//traverse pSorted[attrNo], create splits between pairs of cases w diff response and evaluate on the fly
			
			//parameters that will be changing for different splits
			double volume1 = 0;						//without mv
			double volume2 = nodeV - missV;
			double sum1 = 0;						//without mv
			double sum2 = nodeSum - missSum;

			//parameters of traverse
			bool prevDiff, curDiff; //whether prev or cur attrval had diff response values
			double prevAttrVal; //previous value of the attribute
			double prevResp = QNAN; //response value if same for prev attrval 
			double curAttrVal; //current value of attribute
			double curResp; //current response, if the same

			//parameters of the set of points that moves from one node to the other
			double prevTraV, prevTraSum; //for the previous block
			double curTraV, curTraSum; //for the current block
			prevTraV = 0; prevTraSum = 0;

			dipairv* pSortedVals = &(*pSorted)[attrNo];
			dipairv::iterator pairIt = pSortedVals->begin();
			while(pairIt != pSortedVals->end())
			{//on each iteration of this cycle collect info about the block of cases with the
				//same value of the attribute and if needed, evaluate the split right before it.
				
				//initialize current traverse parameters
				curAttrVal = pairIt->first;
				curResp = (*pItemSet)[pairIt->second].response;
				curDiff = false;
				curTraV = 0;	
				curTraSum = 0;	

				//get next block, update transition parameters
				dipairv::iterator sortedEnd = pSortedVals->end();
				for(;(pairIt != sortedEnd) && (pairIt->first == curAttrVal); pairIt++)
				{
					ItemInfo& item = (*pItemSet)[pairIt->second];
					double coef_sq = item.coef * item.coef;
					curTraV += coef_sq;
					curTraSum += coef_sq * item.response;
					if(!curDiff && (item.response != curResp))
						curDiff = true;
				}

				//if there are different responses in previous and current block 
					//build and evaluate the split between them
				if(!isnan(prevResp) && (prevDiff || curDiff || (prevResp != curResp)))
				{
					newSplits = true;
					//calculate the "short mse" of the new split - parts of sum of se that are different for different splits
					//1. update those parameters that we keep
					volume1 += prevTraV;
					volume2 -= prevTraV;
					sum1 += prevTraSum;
					sum2 -= prevTraSum;

					//2. calculate the short mse of the split from them
					double leftRatio = volume1 / (volume1 + volume2); //does not depend on mv
					double rightRatio = 1 - leftRatio;
						
					double mean1 = (sum1 / leftRatio + missSum) / nodeV;
					double mean2 = (sum2 / rightRatio + missSum) / nodeV;
					double missPred = leftRatio * mean1 + rightRatio * mean2;

					double sqErr1 =  - 2 * mean1 * sum1 + volume1 * mean1 * mean1;
					double sqErr2 = - 2 * mean2 * sum2 + volume2 * mean2 * mean2;
					double missSqErr =  - 2 * missPred * missSum + missV * missPred * missPred; 

					double eval = ( sqErr1 + sqErr2 + missSqErr + squares )/rootVar + penalty;
			
					//evaluate the split point, if it is the best (one of the best) so far, keep it
					if(isnan(bestEval) || (eval < bestEval))
					{
						bestEval = eval;
						bestSplits.clear();
					}
					if(eval == bestEval)
					{
						//create actual split with the split point halfway between attr values
						SplitInfo goodSplit(attr, (curAttrVal + prevAttrVal) / 2, leftRatio);
						bestSplits.push_back(goodSplit);
					}

					//"restart" prev parameters with this block
					prevTraV = curTraV;
					prevTraSum = curTraSum;
				}//end if(!isnan(prevResp) && (prevDiff || curDiff || (prevResp != curResp)))
				else
				{//block was not used, increas "prev" parameters
					prevTraV += curTraV;
					prevTraSum += curTraSum;
				}

				//update previous traverse parameters with values of current
				prevDiff = curDiff;
				prevResp = curResp;
				prevAttrVal = curAttrVal;
			}//end while(pairIt != pSortedVals->end())
		}//end	if(pData->boolAttr(attr))

		//if an attribute is exhausted, delete it, shift to next iteration
		if(!newSplits)
		{
			pAttrs->erase(pAttrs->begin() + attrNo);
			pSorted->erase(pSorted->begin() + attrNo);
		}
		else
			attrNo++;

	}//end for(int attrNo = 0; attrNo < (int)pAttrs->size();)

//choose a random split from those with best mse
	if(!isnan(bestEval))
	{
		int bestSplitN = (int)bestSplits.size();
		int randSplit = rand() % bestSplitN;
		splitting = bestSplits[randSplit];
		// if(attrIds[splitting.divAttr] == 0)
		// 	*numUsed += 1;
		attrIds[splitting.divAttr] = 1;
	}
	return isnan(bestEval);
}

// use the idea of groupTesting and binarySearch. suppose there are s important variable out of d, to get the exact best split
// we need O(n*d) computation. if we use groupTesting and binarySearch to get approximate best split, we only need O(n*slog(s)*log(d/s))


//Chooses and sets best mse split over all attributes and values when no missing values are
// present.
//To compare mse values of splits, we need to calculate only 2 of 3 squared sum components.
//For each attribute it estimates all splits in _one_pass_ over the sorted data, i.e. O(N)
//	(trivial algorithm would produce much less code, but the running time would be O(N^2) )
//Side effect: removes exhausted attributes from the node attribute set
//in:
	// nodeV - size (volume) of the training subset
	// nodeSum - sum of response values in the training subset
//out: true, if best split found. false, if there were no splits

bool CTreeNode::setGroupSplit(double nodeV, double nodeSum, double squares, double rootVar, double mu, int *attrIds, int s, int *numUsed, double *compN)
{	

	double bestEval = QNAN; //current value for the best evaluation
	SplitInfov bestSplits; // all splits that have best (identical) evaluation
	// first evaluate previous used feature and get remaining attrIds to do groupSplit 
	for(int attrNo = 0; attrNo < (int)pAttrs->size(); attrNo++)
	{
		int attr = (*pAttrs)[attrNo];
		// double penalty = (1 - attrIds[attr])*mu; // 0<=mu<1 penalty
		if(attrIds[attr]==1)
		{					//used feature

			dipairv* pSortedVals = &(*pSorted)[attrNo];  // pointer to sorted value and index

			bool newSplits = singleSplit(bestSplits, bestEval, attr, pSortedVals, nodeV, nodeSum, squares, rootVar, 0, compN); //update  bestSplits, bestEval								
			//if an attribute is exhausted, delete it, shift to next iteration
			if(!newSplits)
			{
				pAttrs->erase(pAttrs->begin() + attrNo);
				pSorted->erase(pSorted->begin() + attrNo);
			}
		
		}//end 	for(int attrNo = 0; attrNo < (int)pAttrs->size();)
	}

// Do groupTest among unusedIds
	
	int sampleN = (int)pItemSet->size();
	// cout << "sampleN: " << sampleN << endl;
	// corresponding sorted value and index
	dipairv tmp1(sampleN); 
	dipairv tmp2(sampleN);
	dipairv* Ptmp1 = &tmp1;
	dipairv* Ptmp2 = &tmp2;

	// binary search

	int st = 0;
	int ed = pData->getAttrN() - 1;
	while(st<ed)
	{
		int m = (st+ed)/2;
		double groupSplitVal1 = QNAN; // split eval for group of variables from subset[st] to subset[m] 
		double groupSplitVal2 = QNAN; // split eval for group of variables from subset[m+1] to subset[ed]
		bool split1 = false;
		bool split2 = false;

		for(int sampleNo = 0; sampleNo < sampleN; sampleNo++)
		{
			int caseNo = (*pItemSet)[sampleNo].key;
			double value1 = pData->getRangeSum(caseNo, st, m);
			double value2 = pData->getRangeSum(caseNo, m+1, ed);
			// cout << "caseNo: " << caseNo << endl;
			// cout << "value1: " << value1 << endl;
			// cout << "value2: " << value2 << endl;
			(*Ptmp1)[sampleNo] = dipair(value1, sampleNo);
			(*Ptmp2)[sampleNo] = dipair(value2, sampleNo);
		}
		sort(Ptmp1->begin(), Ptmp1->end());
		sort(Ptmp2->begin(), Ptmp2->end());
		*compN += (double)(sampleN + sampleN * max(1.0,log(sampleN)))/(double)pData->getTrainN();

		 // cout << "st: "<<st << " m " <<m << " ed: "<<ed << endl;

		if( (st==m) && pData->isActive(m) )
		{
			split1 = singleSplit(bestSplits, bestEval, m, Ptmp1, nodeV, nodeSum, squares, rootVar, mu, compN); //update  bestSplits, bestEval

		}
		else
			split1 = singleSplit(bestSplits, groupSplitVal1, -1, Ptmp1, nodeV, nodeSum, squares, rootVar, mu, compN); //do not update  bestSplits, bestEval

		if( (ed==m+1) && pData->isActive(m+1) )
		{
			split2 = singleSplit(bestSplits, bestEval, m+1, Ptmp2, nodeV, nodeSum, squares, rootVar, mu, compN); //update  bestSplits, bestEval
		}
		else
			split2 = singleSplit(bestSplits, groupSplitVal2, -1, Ptmp2, nodeV, nodeSum, squares, rootVar, mu, compN); //do not update  bestSplits, bestEval

		 // cout << "splitval1: " << groupSplitVal1 << "splitval2: " << groupSplitVal2 << endl;
		 // cout << "split1: " << split1 << "split2: " << split2 << endl;

		if(!split2 || (split1 && ( groupSplitVal1 < groupSplitVal2)))
			ed = m;
		else
			st = m+1;


	}


	



	//choose a random split from those with best mse
	if(!isnan(bestEval))
	{
		int bestSplitN = (int)bestSplits.size();
		int randSplit = rand() % bestSplitN;
		splitting = bestSplits[randSplit];
		if(pData->boolAttr(splitting.divAttr))
		{//one can split only once on boolean attribute, remove it from the set of attributes
			int attrNo = erasev(pAttrs, splitting.divAttr);
			pSorted->erase(pSorted->begin() + attrNo);	
				//it is an empty vector (the attribute is boolean), but we still need to remove it
		}
		if(attrIds[splitting.divAttr] == 0)
			*numUsed += 1;

		attrIds[splitting.divAttr] = 1;
	}
	return isnan(bestEval);
}


bool CTreeNode::singleSplit(SplitInfov& bestSplits, double& bestEval, int attr, dipairv* pSortedVals, double nodeV, double nodeSum, double squares, double rootVar, double mu, double* compN)
{

	
		bool newSplits = false;	//true if any splits were added for this attribute
		if( (attr > -1) && (pData->boolAttr(attr)))	
		{//boolean attribute
			//there is exactly one split for a boolean attribute, evaluate it
			SplitInfo boolSplit(attr, 0.5);
			double eval = evalBool(boolSplit, nodeV, nodeSum, squares, rootVar) + mu;
			if(!isnan(eval))

			{

			newSplits = true;

			//save if this is one of the best splits
				if(isnan(bestEval) || (eval < bestEval))
				{
					bestEval = eval;
					bestSplits.clear();
				}
				if(eval == bestEval)
					bestSplits.push_back(SplitInfo(boolSplit));
			}
		}
		else //continuous attribute or groupTest
		{//candidate splits are installed between all pairs of neighbour values (values are sorted)
		 //all splits are calculated in one pass

		
			//traverse pSorted[attrNo], create crisp splits between pairs of cases w diff response
				//and evaluate on the fly
			
			//parameters that will be changing for different splits
			double volume1 = 0;						
			double volume2 = nodeV;
			double sum1 = 0;						
			double sum2 = nodeSum;

			//parameters of traverse
			bool prevDiff, curDiff; //whether prev or cur attrval had diff response values
			double prevAttrVal; //previous value of the attribute
			double prevResp = QNAN; //response value if same for prev attrval 
			double curAttrVal; //current value of attribute
			double curResp; //current response, if the same

			//parameters of the set of points that moves from one node to the other
			double prevTraV, prevTraSum; //for the previous block
			double curTraV, curTraSum; //for the current block
			prevTraV = 0; prevTraSum = 0;

			// dipairv* pSortedVals = &(*pSorted)[attrNo];  // pointer to sorted value and index
			dipairv::iterator pairIt = pSortedVals->begin(); // iterator for that pair
			while(pairIt != pSortedVals->end())
			{//on each iteration of this cycle collect info about the block of cases with the
				//same value of the attribute and if needed, evaluate the split right before it.
				// cout<< pairIt->first << endl;
				//initialize current traverse parameters
				curAttrVal = pairIt->first;
				curResp = (*pItemSet)[pairIt->second].response;
				curDiff = false;
				curTraV = 0;	
				curTraSum = 0;	

				//get next block, update transition parameters
				dipairv::iterator sortedEnd = pSortedVals->end();
				for(;(pairIt != sortedEnd) && (pairIt->first == curAttrVal); pairIt++)
				{
					ItemInfo& item = (*pItemSet)[pairIt->second];
					curTraV ++;
					curTraSum += item.response;
					if(!curDiff && (item.response != curResp))
						curDiff = true;
				}

				//if there are different responses in previous and current block 
					//build and evaluate the split between them
				if(!isnan(prevResp) && (prevDiff || curDiff || (prevResp != curResp)))
				{
					newSplits = true;
					//calculate the "short mse" of the new split - parts of sum of se that are different for different splits
					//1. update those parameters that we keep
					volume1 += prevTraV;
					volume2 -= prevTraV;
					sum1 += prevTraSum;
					sum2 -= prevTraSum;

					//2. calculate short mse of the split from them
					double leftRatio = volume1 / nodeV;

					double mean1 = sum1 / volume1;
					double mean2 = sum2 / volume2;

					double sqErr1 =  - mean1 * sum1;
					double sqErr2 =  - mean2 * sum2;

					double eval = ( sqErr1 + sqErr2 + squares )/rootVar + mu;
			
					//evaluate the split point, if it is the best (one of the best) so far, keep it
					if(isnan(bestEval) || (eval < bestEval))
					{
						bestEval = eval;
						if(attr > -1) // only update bestSplits if this is a variable split
							bestSplits.clear();
					}
					if(eval == bestEval)
					{
						if(attr > -1) // only update bestSplits if this is a variable split
						{
						//create actual split with the split point halfway between attr values
						SplitInfo goodSplit(attr, (curAttrVal + prevAttrVal) / 2, leftRatio);
						bestSplits.push_back(goodSplit);
						}
					}

					//"restart" prev parameters with this block
					prevTraV = curTraV;
					prevTraSum = curTraSum;
				}//end if(!isnan(prevResp) && (prevDiff || curDiff || (prevResp != curResp)))
				else
				{//block was not used, increas "prev" parameters
					prevTraV += curTraV;
					prevTraSum += curTraSum;
				}

				//update previous traverse parameters with values of current
				prevDiff = curDiff;
				prevResp = curResp;
				prevAttrVal = curAttrVal;
			}//end while(pairIt != pSortedVals->end())				
	}//end if continuous
*compN += (double)pSortedVals->size()/(double)pData->getTrainN();

return newSplits; 
}






//Calculates short sum of squared errors of the boolean split for the data without missing values. 
//Does not require sorting.
//parameters: 
//	in-out: canSplit - info about the splitting being evaluated
//	in: nodeV - size (volume) of the node train subset
//  in: nodeSum - sum of responses of the cases in node train subset
double CTreeNode::evalBool(SplitInfo& canSplit, double nodeV, double nodeSum, double squares, double rootVar)
{
	double volume1 = 0;
	double sum1 = 0;

	for(ItemInfov::iterator itemIt = pItemSet->begin(); itemIt != pItemSet->end(); itemIt++)
	{
		double value = pData->getValue(itemIt->key, canSplit.divAttr, TRAIN);
		if(value == 0) //left
		{
			volume1++;
			sum1 += itemIt->response;
		}

	}
	double volume2 = nodeV - volume1;
	double sum2 = nodeSum - sum1;

	//if either of volumes is zero, the split is not valid
	if((volume1 == 0) || (volume2 == 0))
		return QNAN;

	double mean1 = sum1 / volume1;
	double mean2 = sum2 / volume2;
	double sqErr1 =  - mean1 * sum1 ;
	double sqErr2 =  - mean2 * sum2 ;

	canSplit.missingL = volume1 / nodeV; 

	return (sqErr1 + sqErr2 + squares)/rootVar;
}

//Calculates short sum of squared errors of the boolean split for the data with missing values. Does not require sorting.
//Missing values: if value of the attribute is missing, the case goes to both branches with 
//	coefficients proportional to the volumes of cases with defined value of the attributes going to
//	the same branches. All volumes and sums in the nodes are calculated using squared coefficients
//parameters: 
//	in-out: canSplit - info about the splitting being evaluated
//	in: nodeV - size (volume) of the whole node train subset (calc. with sq coef)
//  in: nodeSum - sum of responses of the cases in node train subset (calc. with sq coef)
//	in: missV - volume of the data points with missing values  (calc. with sq coef)
//  in: missSum - sum of responses data points with missing values (calc. with sq coef)
double CTreeNode::evalBoolMV(SplitInfo& canSplit, double nodeV, double nodeSum, double squares, double rootVar, double missV, double missSum)
{
	double volume1 = 0;
	double sum1 = 0;

	for(ItemInfov::iterator itemIt = pItemSet->begin(); itemIt != pItemSet->end(); itemIt++)
	{
		double value = pData->getValue(itemIt->key, canSplit.divAttr, TRAIN);
		double coef_sq = itemIt->coef * itemIt->coef;
		double& resp = itemIt->response;
		if(!isnan(value) && (value == 0))	//not missing, left
		{
			volume1 += coef_sq;
			sum1 += resp * coef_sq;
		}
	}

	double volume2 = nodeV - missV - volume1;
	double sum2 = nodeSum - missSum - sum1;

	//if either of volumes is zero, the split is not valid
	if((volume1 - 0.0000000001 <= 0) || (volume2 - 0.0000000001 <= 0))
		return QNAN;

	double leftRatio = volume1 / (volume1 + volume2); //does not depend on mv
	double rightRatio = 1 - leftRatio;
	
	double mean1 = (sum1 / leftRatio + missSum) / nodeV;
	double mean2 = (sum2 / rightRatio + missSum) / nodeV;
	double missPred = leftRatio * mean1 + rightRatio * mean2;

	double sqErr1 =  - 2 * mean1 * sum1 + volume1 * mean1 * mean1;
	double sqErr2 =  - 2 * mean2 * sum2 + volume2 * mean2 * mean2;
	double missSqErr = - 2 * missPred * missSum + missV * missPred * missPred; 

	canSplit.missingL = leftRatio;

	return (sqErr1 + sqErr2 + missSqErr + squares)/rootVar;
}


//dumps the node contents into a binary file
//links to other nodes are not saved, the tree will be reconstructed from the order of nodes 
void CTreeNode::save(fstream& fsave)
{
	//first bit indicates whether the node is a leaf: required for reconstructing tree structure
	bool leaf = isLeaf();
	fsave.write((char*) &leaf, sizeof(bool));
	
	if(leaf)	//save node prediction
		fsave.write((char*) &((*pItemSet)[0].response), sizeof(double));
	else		
	{			//save splitting 
		fsave.write((char*) &(splitting.divAttr), sizeof(int));
		fsave.write((char*) &(splitting.border), sizeof(double));
		fsave.write((char*) &(splitting.missingL), sizeof(double));
	}
	if(fsave.bad() || fsave.fail())
		throw TREE_WRITE_ERR;
}

//loads the node from a binary file
//returns true if the node is a leaf
bool CTreeNode::load(fstream& fload)
{
	del();

	bool leaf = false;
	fload.read((char*)&leaf, sizeof(bool));

	if(leaf)	//load node prediction
	{
		double prediction;
		fload.read((char*) &prediction, sizeof(double));
		makeLeaf(prediction);
	}
	else		
	{			//load splitting 
		fload.read((char*) &(splitting.divAttr), sizeof(int));
		fload.read((char*) &(splitting.border), sizeof(double));
		fload.read((char*) &(splitting.missingL), sizeof(double));
	}
	if(fload.fail())
		throw TREE_LOAD_ERR;
	if(!leaf && !pData->isActive(splitting.divAttr))
		throw MODEL_ATTR_MISMATCH_ERR;
	return leaf;
}

//delete an attribute from internal node structures
void CTreeNode::delAttr(int attrNo)
{
	int localNo = -1;
	erasev(pAttrs, attrNo, localNo);
	pSorted->erase(pSorted->begin() + localNo);
}
