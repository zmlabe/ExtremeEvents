"""
Functions are useful untilities for interpretation of ANN
 
Notes
-----
    Author : Zachary Labe
    Date   : 12 January 2021
    
Usage
-----
    [1] deepTaylorAnalysis(model,XXt,YYt,biasBool,annType,num_of_class,yearlabels)
"""

###############################################################################
###############################################################################
###############################################################################

def deepTaylorAnalysis(model,XXt,YYt,biasBool,annType,num_of_class,yearlabels):
    """
    Calculate Deep Taylor for LRP
    """
    print('<<<< Started deepTaylorAnalysis() >>>>')
    
    ### Import modules
    import numpy as np 
    import innvestigate
    
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ### Create the innvestigate analyzer instance for each sample
    if(annType=='class'):
        model_nosoftmax = innvestigate.utils.model_wo_softmax(model)
    analyzer = innvestigate.analyzer.relevance_based.relevance_analyzer.LRPAlphaBeta(
                                model_nosoftmax,alpha=1,beta=0,bias=biasBool)

    deepTaylorMaps = np.empty(np.shape(XXt))
    deepTaylorMaps[:] = np.nan

    # analyze each input via the analyzer
    for i in np.arange(0,np.shape(XXt)[0]):
        sample = XXt[i]
        analyzer_output = analyzer.analyze(sample[np.newaxis,...])
        deepTaylorMaps[i] = analyzer_output/np.sum(analyzer_output.flatten())

    print('done with Deep Taylor analyzer normalization')     
    
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ### Compute the frequency of data at each point and the average relevance 
    ### normalized by the sum over the area and the frequency above the 90th 
    ### percentile of the map
    summaryDTq = np.reshape(deepTaylorMaps,(deepTaylorMaps.shape[0]//len(yearlabels),
                                            len(yearlabels),deepTaylorMaps.shape[1]))
    # summaryDT = np.nanmean(summaryDTq,axis=0)
    
    print('<<<< Completed deepTaylorAnalysis() >>>>')    
    return(summaryDTq)