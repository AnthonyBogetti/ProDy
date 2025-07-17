# -*- coding: utf-8 -*-
"""This module defines functions for searching the Dali server and processing PDB structural ensembles. (similar and diverse ensembles)"""

import re
import numpy as np
from prody.atomic import Atomic, AtomGroup, AtomMap
from prody.proteins.pdbfile import _getPDBid
from prody.measure import getRMSD, getTransformation
from prody.utilities import checkCoords, checkWeights, createStringIO
from prody import LOGGER, PY3K
from prody import parsePDB, writePDBStream
# if PY3K:
    # import urllib.parse as urllib
    # import urllib.request as urllib2
# else:
    # import urllib
    # import urllib2
from prody.ensemble import Ensemble
from prody.ensemble import PDBEnsemble
import os

#__all__ = ['DaliRecor', 'searchDal', 'daliFilterMultimer', 'daliFilterMultimers']

def searchDal(pdb, chain=None, subset='fullPDB', daliURL=None, **kwargs):
    """Search Dali server with input of PDB ID (or local PDB file) and chain ID.
    Dali server: http://ekhidna2.biocenter.helsinki.fi/dali/
    
    :arg pdb: PDB code or local PDB file for the protein to be searched

    :arg chain: chain identifier (only one chain can be assigned for PDB)
    :type chain: str

    :arg subset: fullPDB, PDB25, PDB50, PDB90
    :type subset: str
    """
    
    import requests
    
    LOGGER.timeit('_dali')
    # timeout = 120
    timeout = kwargs.pop('timeout', 120)
    
    if daliURL is None:
        daliURL = "http://ekhidna2.biocenter.helsinki.fi/cgi-bin/sans/dump.cgi"
    
    if isinstance(pdb, Atomic):
        atoms = pdb
        chain_set = set(atoms.getChids())
        if chain and not chain in chain_set:
            raise ValueError('input structure (%s) does not have chain %s'%(atoms.getTitle(), chain))
        
        if len(chain_set) > 1:
            if not chain:
                raise TypeError('the structure (%s) contains more than one chain, therefore a chain identifier '
                                'needs to be specified'%pdb.getTitle())
            atoms = atoms.select('chain '+chain)
        else:
            chain = chain_set.pop()
            
        stream = createStringIO()
        writePDBStream(stream, atoms)
        data = stream.getvalue()
        stream.close()
        files = {"file1" : data}

        pdbId = atoms.getTitle()
        pdb_chain = ''
        dali_title = 'Title_'+pdbId+chain
    elif isinstance(pdb, str):
        if os.path.isfile(pdb):
            atoms = parsePDB(pdb)
            chain_set = set(atoms.getChids())
            # pdbId = "s001"
            filename = os.path.basename(pdb)
            filename, ext = os.path.splitext(filename)
            if ext.lower() == '.gz':
                filename2, ext2 = os.path.splitext(filename)
                if ext2.lower() == '.pdb':
                    filename = filename2
            pdbId = filename
            if chain and not chain in chain_set:
                raise ValueError('input PDB file does not have chain ' + chain)
            
            if len(chain_set) > 1:
                if not chain:
                    raise TypeError('PDB file (%s) contains more than one chain, therefore a chain identifier '
                                    'needs to be specified'%pdb)
                atoms = atoms.select('chain '+chain)
                #local_temp_pdb = pdbId+chain+'.pdb'
                #local_temp_pdb = 's001'+chain+'.pdb'
                stream = createStringIO()
                writePDBStream(stream, atoms)
                data = stream.getvalue()
                stream.close()
            else:
                data = open(pdb, "rb")
                chain = chain_set.pop()
            files = {"file1" : data}
            # case: multiple chains.             apply fetch ? multiple times?
            pdb_chain = ''
            dali_title = 'Title_' + pdbId + chain
        else:
            pdbId, ch = _getPDBid(pdb)
            if not chain:
                chain = ch
            if not chain:
                raise TypeError('a chain identifier is needed for the search')
            pdb_chain = pdbId + chain
            dali_title = 'Title_' + pdb_chain
            files = ''
    parameters = { 'cd1' : pdb_chain, 'method': 'search', 'title': dali_title, 'address': '' }
    # enc_params = urllib.urlencode(parameters).encode('utf-8')
    # request = urllib2.Request(daliURL, enc_params)
    request = requests.post(daliURL, parameters, files=files)
    try_error = 3
    while try_error >= 0:
        try:
            # url = urllib2.urlopen(request).url
            url = request.url
            break
        except:
            try_error -= 1
            if try_error >= 0:
                LOGGER.sleep(2, '. Connection error happened. Trying to reconnect...')
                continue
            else:
                # url = urllib2.urlopen(request).url
                url = request.url
                break
    if url.split('.')[-1].lower() in ['html', 'php']:
        # print('test -1: '+url)
        url = url.replace(url.split('/')[-1], '')
    LOGGER.debug('Submitted Dali search for PDB "{0}{1}".'.format(pdbId, chain))
    LOGGER.info(url)
    LOGGER.clear()
    
    return DaliRecor(url, pdbId, chain, subset=subset, timeout=timeout, **kwargs)
    

class DaliRecor(object):

    """A class to store results from Dali PDB search."""

    def __init__(self, url, pdbId, chain, subset='fullPDB', localFile=False, **kwargs):
        """Instantiate a DaliRecord object instance.

        :arg url: url of Dali results page or local dali results file
        :arg pdbId: PDB code for searched protein
        :arg chain: chain identifier (only one chain can be assigned for PDB)
        :arg subset: fullPDB, PDB25, PDB50, PDB90. Ignored if localFile=True (url is a local file)
        :arg localFile: whether provided url is a path for a local dali results file
        """

        self._url = url
        self._pdbId = pdbId
        self._chain = chain
        subset = subset.upper()
        if subset == "FULLPDB" or subset not in ["PDB25", "PDB50", "PDB90"]:
            self._subset = ""
        else:
            self._subset = "-"+subset[3:]
        timeout = kwargs.pop('timeout', 120)

        self._title = pdbId + '-' + chain
        self._alignPDB = None
        self._filterDict = None
        self._max_index = None
        self.fetch(self._url, localFile=localFile, timeout=timeout, **kwargs)

    def fetch(self, url=None, localFile=False, **kwargs):
        """Get Dali record from url or file.

        :arg url: url of Dali results page or local dali results file
            If None then the url already associated with the DaliRecord object is used.
        :type url: str

        :arg localFile: whether provided url is a path for a local dali results file
        :type localFile: bool

        :arg timeout: amount of time until the query times out in seconds
            default value is 120
        :type timeout: int

        :arg localfolder: folder in which to find the local file
            default is the current folder
        :type localfolder: str
        """
        if localFile:
            dali_file = open(url, 'r')
            data = dali_file.read()
            dali_file.close()
        else:
            import requests
            
            if url == None:
                url = self._url
            
            sleep = 2
            timeout = kwargs.pop('timeout', 120)
            LOGGER.timeit('_dali')
            log_message = ''
            try_error = 3
            while True:
                LOGGER.write('Connecting to Dali for search results...')
                LOGGER.clear()
                try:
                    # html = urllib2.urlopen(url).read()
                    html = requests.get(url).content
                except:
                    try_error -= 1
                    if try_error >= 0:
                        LOGGER.sleep(2, '. Connection error happened. Trying to reconnect...')
                        continue
                    else:
                        # html = urllib2.urlopen(url).read()
                        html = requests.get(url).content
                if PY3K:
                    html = html.decode()
                if html.find('Status: Queued') > -1:
                    log_message = '(Dali search is queued)...'
                elif html.find('Status: Running') > -1:
                    log_message = '(Dali search is running)...'
                elif html.find('Your job') == -1 and html.find('.txt') > -1:
                    break
                elif html.find('ERROR:') > -1:
                    LOGGER.warn(': Dali search reported an ERROR!')
                    self.isSuccess = False
                    return False
                sleep = 20 if int(sleep * 1.5) >= 20 else int(sleep * 1.5)
                if LOGGER.timing('_dali') > timeout:
                    LOGGER.warn(': Dali search has timed out. \nThe results can be obtained later using the fetch() method.')
                    self.isSuccess = False
                    return False
                LOGGER.sleep(int(sleep), 'to reconnect to Dali '+log_message)
                LOGGER.clear()
            LOGGER.clear()
            LOGGER.report('Dali results were fetched in %.1fs.', '_dali')
            lines = html.strip().split('\n')
            file_name = re.search('=.+-90\\.txt', html).group()[1:]
            file_name = file_name[:-7]
            # LOGGER.info(url+file_name+self._subset+'.txt')
            # data = urllib2.urlopen(url+file_name+self._subset+'.txt').read()
            data = requests.get(url+file_name+self._subset+'.txt').content
            if PY3K:
                data = data.decode()
            localfolder = kwargs.pop('localfolder', '.')

            if file_name.lower().startswith('s001'):
                temp_name = self._pdbId + self._chain
            else:
                temp_name = file_name
            temp_name += self._subset + '_dali.txt'
            if localfolder != '.' and not os.path.exists(localfolder):
                os.mkdir(localfolder)
            with open(localfolder+os.sep+temp_name, "w") as file_temp: file_temp.write(html + '\n' + url+file_name+self._subset+'.txt' + '\n' + data)
            # with open(temp_name, "a+") as file_temp: file_temp.write(url+file_name + '\n' + data)
        data_list = data.strip().split('# ')
        # No:  Chain   Z    rmsd lali nres  %id PDB  Description -> data_list[3]
        # Structural equivalences -> data_list[4]
        # Translation-rotation matrices -> data_list[5]
        map_temp_dict = dict()
        lines = data_list[4].strip().split('\n')
        self._lines_4 = lines
        mapping_temp = np.genfromtxt(lines[1:], delimiter = (4,1,14,6,2,4,4,5,2,4,4,3,5,4,3,5,6,3,5,4,3,5,28), 
                                     usecols = [0,3,5,7,9,12,15,15,18,21], dtype='|i4')
        # [0,3,5,7,9,12,15,15,18,21] -> [index, residue_a, residue_b, residue_i_a, residue_i_b, resid_a, resid_b, resid_i_a, resid_i_b]
        for map_i in mapping_temp:
            if not map_i[0] in map_temp_dict:
                map_temp_dict[map_i[0]] = [[map_i[1], map_i[2], map_i[3], map_i[4]]]
            else:
                map_temp_dict[map_i[0]].append([map_i[1], map_i[2], map_i[3], map_i[4]])
        self._max_index = max(mapping_temp[:,2])
        self._mapping = map_temp_dict
        self._data = data_list[3]
        lines = data_list[3].strip().split('\n')
        # daliInfo = np.genfromtxt(lines[1:], delimiter = (4,3,6,5,5,5,6,5,57), usecols = [0,2,3,4,5,6,7,8], 
                                # dtype=[('id', '<i4'), ('pdb_chain', '|S6'), ('Z', '<f4'), ('rmsd', '<f4'), 
                                # ('len_align', '<i4'), ('nres', '<i4'), ('identity', '<i4'), ('title', '|S70')])
        daliInfo = np.genfromtxt(lines[1:], delimiter = (4,3,6,5,5,5,6,5,57), usecols = [0,2,3,4,5,6,7,8], 
                                dtype=[('id', '<i4'), ('pdb_chain', '|U6'), ('Z', '<f4'), ('rmsd', '<f4'), 
                                ('len_align', '<i4'), ('nres', '<i4'), ('identity', '<i4'), ('title', '|U70')])
        if daliInfo.ndim == 0:
            daliInfo = np.array([daliInfo])
        pdbListAll = []
        self._daliInfo = daliInfo
        dali_temp_dict = dict()
        for temp in self._daliInfo:
            temp_dict = dict()
            pdb_chain = temp[1].strip()[0:6]
            # U6 and U70 were used as the dtype for np.genfromtext -> unicode string were used in daliInfo 
            # if PY3K:
                # pdb_chain = pdb_chain.decode()
            pdb_chain = str(pdb_chain)
            temp_dict['pdbId'] = pdbid = pdb_chain[0:4].lower()
            temp_dict['chainId'] = chid = pdb_chain[5:6]
            temp_dict['pdb_chain'] = pdb_chain = pdbid + chid
            temp_dict['Z'] = temp[2]
            temp_dict['rmsd'] = temp[3]
            temp_dict['len_align'] = temp[4]
            temp_dict['nres'] = temp[5]
            temp_dict['identity'] = temp[6]
            temp_dict['mapping'] = (np.array(map_temp_dict[temp[0]])-1).tolist()
            temp_dict['map_ref'] = [x for map_i in (np.array(map_temp_dict[temp[0]])-1).tolist() for x in range(map_i[0], map_i[1]+1)]
            temp_dict['map_sel'] = [x for map_i in (np.array(map_temp_dict[temp[0]])-1).tolist() for x in range(map_i[2], map_i[3]+1)]
            dali_temp_dict[pdb_chain] = temp_dict
            pdbListAll.append(pdb_chain)
        self._pdbListAll = tuple(pdbListAll)
        self._pdbList = self._pdbListAll
        self._alignPDB = dali_temp_dict
        LOGGER.info('Obtained ' + str(len(pdbListAll)) + ' PDB chains from Dali for '+self._pdbId+self._chain+'.')
        self.isSuccess = True
        return True
        
    def getPDBs(self, filtered=True):
        """Returns PDB list (filters may be applied)"""

        if self._alignPDB is None:
            LOGGER.warn('Dali Record does not have any data yet. Please run fetch.')
        
        if filtered:
            return self._pdbList
        return self._pdbListAll
        
    def getHits(self):
        """Returns the dictionary associated with the DaliRecord"""

        if self._alignPDB is None:
            LOGGER.warn('Dali Record does not have any data yet. Please run fetch.')

        return self._alignPDB
        
    def getFilterList(self):
        """Returns a list of PDB IDs and chains for the entries that were filtered out"""
        
        filterDict = self._filterDict
        if filterDict is None:
            raise ValueError('You cannot obtain the list of filtered out entries before doing any filtering.')

        temp_str = ', '.join([str(len(filterDict['len'])), str(len(filterDict['rmsd'])), 
                            str(len(filterDict['Z'])), str(len(filterDict['identity']))])
        LOGGER.info('Filtered out [' + temp_str + '] for [length, RMSD, Z, identity]')
        return self._filterList

    
    def getMapping(self, key):
        """Get mapping for a particular entry in the DaliRecord"""

        if self._alignPDB is None:
            LOGGER.warn('Dali Record does not have any data yet. Please run fetch.')
            return None
        
        try:
            info = self._alignPDB[key]
            mapping = [info['map_ref'], info['map_sel']]
        except KeyError:
            return None
        return mapping

    def getMappings(self):
        """Get all mappings in the DaliRecord"""

        if self._alignPDB is None:
            LOGGER.warn('Dali Record does not have any data yet. Please run fetch.')
            return None

        map_dict = {}
        for key in self._alignPDB:
            info = self._alignPDB[key]
            mapping = [info['map_ref'], info['map_sel']]
            map_dict[key] = mapping
        return map_dict

    mappings = property(getMappings)

    def filter(self, cutoff_len=None, cutoff_rmsd=None, cutoff_Z=None, cutoff_identity=None):
        """Filters out PDBs from the PDBList and returns the PDB list.
        PDBs that satisfy any of the following criterion will be filtered out.
        (1) Length of aligned residues < cutoff_len (must be an integer or a float between 0 and 1);
        (2) RMSD < cutoff_rmsd (must be a positive number);
        (3) Z score < cutoff_Z (must be a positive number);
        (4) Identity > cutoff_identity (must be an integer or a float between 0 and 1).
        """
        if self._max_index is None:
            LOGGER.warn('DaliRecord has no data. Please use the fetch() method.')
            return None

        if cutoff_len == None:
            # cutoff_len = int(0.8*self._max_index)
            cutoff_len = 0.00000001
        elif not isinstance(cutoff_len, (float, int)):
            raise TypeError('cutoff_len must be a float or an integer')
        elif cutoff_len <= 1 and cutoff_len > 0:
            cutoff_len = int(cutoff_len*self._max_index)
        elif cutoff_len <= self._max_index and cutoff_len > 0:
            cutoff_len = int(cutoff_len)
        else:
            raise ValueError('cutoff_len must be a float between 0 and 1, or an int not greater than the max length')
            
        if cutoff_rmsd == None:
            cutoff_rmsd = 1000
        elif not isinstance(cutoff_rmsd, (float, int)):
            raise TypeError('cutoff_rmsd must be a float or an integer')
        elif cutoff_rmsd >= 0:
            cutoff_rmsd = float(cutoff_rmsd)
        else:
            raise ValueError('cutoff_rmsd must be a number not less than 0')
            
        if cutoff_Z == None:
            cutoff_Z = 0
        elif not isinstance(cutoff_Z, (float, int)):
            raise TypeError('cutoff_Z must be a float or an integer')
        elif cutoff_Z >= 0:
            cutoff_Z = float(cutoff_Z)
        else:
            raise ValueError('cutoff_Z must be a number not less than 0')
            
        if cutoff_identity == None or cutoff_identity == 0:
            cutoff_identity = 0
        elif not isinstance(cutoff_identity, (float, int)):
            raise TypeError('cutoff_identity must be a float or an integer')
        elif cutoff_identity <= 1 and cutoff_identity > 0:
            cutoff_identity = float(cutoff_identity*100)
        elif cutoff_identity <= 100 and cutoff_identity > 0:
            cutoff_identity = float(cutoff_identity)
        else:
            raise ValueError('cutoff_identity must be a float between 0 and 1, or a number between 0 and 100')
            
        # debug:
        # print('cutoff_len: ' + str(cutoff_len) + ', ' + 'cutoff_rmsd: ' + str(cutoff_rmsd) + ', ' + 'cutoff_Z: ' + str(cutoff_Z) + ', ' + 'cutoff_identity: ' + str(cutoff_identity))
        
        daliInfo = self._alignPDB
        if daliInfo is None:
            raise ValueError("Dali Record does not have any data yet. Please run fetch.")

        pdbListAll = self._pdbListAll
        missing_ind_dict = dict()
        ref_indices_set = set(range(self._max_index))
        filterListLen = []
        filterListRMSD = []
        filterListZ = []
        filterListIdentity = []
        
        RMSDs = []
        
        # keep the first PDB (query PDB)
        for pdb_chain in pdbListAll[1:]:
            temp_dict = daliInfo[pdb_chain]
            # print ('currRMSD', temp_dict['rmsd'])
            # filter: len_align, identity, rmsd, Z
            RMSDs.append(temp_dict['rmsd'])
            if temp_dict['len_align'] < cutoff_len:
                # print('Filter out ' + pdb_chain + ', len_align: ' + str(temp_dict['len_align']))
                filterListLen.append(pdb_chain)
                continue
            if temp_dict['rmsd'] > cutoff_rmsd:
                # print('Filter out ' + pdb_chain + ', rmsd: ' + str(temp_dict['rmsd']))
                filterListRMSD.append(pdb_chain)
                # print ('currRMSD', temp_dict['rmsd'])
                # print(pdb_chain)
                continue
            if temp_dict['Z'] < cutoff_Z:
                # print('Filter out ' + pdb_chain + ', Z: ' + str(temp_dict['Z']))
                filterListZ.append(pdb_chain)
                continue
            if temp_dict['identity'] < cutoff_identity:
                # print('Filter out ' + pdb_chain + ', identity: ' + str(temp_dict['identity']))
                filterListIdentity.append(pdb_chain)
                continue
            temp_diff = list(ref_indices_set - set(temp_dict['map_ref']))
            for diff_i in temp_diff:
                if not diff_i in missing_ind_dict:
                    missing_ind_dict[diff_i] = 1
                else:
                    missing_ind_dict[diff_i] += 1
        self._missing_ind_dict = missing_ind_dict
        
        filterList = filterListLen + filterListRMSD + filterListZ + filterListIdentity
        filterDict = {'len': filterListLen, 'rmsd': filterListRMSD, 'Z': filterListZ, 'identity': filterListIdentity}
        self._filterList = filterList
        self._filterDict = filterDict
        self._pdbList = [self._pdbListAll[0]] + [item for item in self._pdbListAll[1:] if not item in filterList]
        LOGGER.info(str(len(filterList)) + ' PDBs have been filtered out from '+str(len(pdbListAll))+' Dali hits (remaining: '+str(len(pdbListAll)-len(filterList))+').')
        
        filterRMSD = []
        diverseRMSD = []
        for item in RMSDs:
            if item < cutoff_rmsd:
                filterRMSD.append(item)
            if item > 1.0:
                diverseRMSD.append(item)
        
        LOGGER.info ('RMSD less than ' + str(cutoff_rmsd) + ',' + str(mean(filterRMSD)) + '±' + str(std(filterRMSD)))
        LOGGER.info ('RMSD greater than 1A ' + str(mean(diverseRMSD)) + '±' + str(std(diverseRMSD)))
        
        return self._pdbList
    
    def getTitle(self):
        """Return the title of the record"""

        return self._title

def daliFilterMultimer(atoms, dali_rec, n_chains=None):
    """
    Filters multimers to only include chains with Dali mappings.

    :arg atoms: the multimer to be filtered
    :type atoms: :class:`.Atomic`

    :arg dali_rec: the DaliRecord object with which to filter chains
    :type dali_rec: :class:`.DaliRecord`
    """
    if not isinstance(atoms, Atomic):
        raise TypeError("atoms should be an Atomic object")

    if not isinstance(dali_rec, DaliRecord):
        raise TypeError("dali_rec should be a DaliRecord")
    try:
        keys = dali_rec._alignPDB
    except:
        raise AttributeError("Dali Record does not have any data yet. Please run fetch.")

    numChains = 0
    atommap = None
    for i, chain in enumerate(atoms.iterChains()):
        m = dali_rec.getMapping(chain.getTitle()[:4] + chain.getChid())
        if m is not None:
            numChains += 1
            if atommap is None:
                atommap = chain
            else:
                atommap += chain

    if n_chains is None or numChains == n_chains:
        return atommap
    else:
        return None

def daliFilterMultimers(structures, dali_rec, n_chains=None):
    """A wrapper for daliFilterMultimer to apply to multiple structures.
    """
    dali_ags = []
    for entry in structures:
        result = daliFilterMultimer(entry, dali_rec, n_chains)
        if result is not None:
            dali_ags.append(result)
    return dali_ags
    
    
# ************************************************************************************************
# This section used SignDy to build homologous ensembles
from numpy import *
from prody import *
from matplotlib.pyplot import *

def getModesGivenThreshold(eigenVals, thereshold):
    """
    Check how many modes are supposed to be included given the contribution.

    :eigenVals: eigenvalues from the GNM modes
    :thereshold: 0-1, contributions

    :mode: return # of modes that reach to the contributions
    """
    contribution = 0
    mode = 0
    while contribution <= thereshold:
        mode += 1
        contribution = getContribution(eigenVals, mode)
    return mode

def savePDBList(pdb_list, currPDB, chain):
    """
    Saves a list of PDB IDs and chains in a 'Results' folder.

    :param pdb_list: List of PDB chain names (e.g., ['3d4sA', '3ny9A']).
    :type pdb_list: list of str

    :param currPDB: Name of the current PDB (used to name the file).
    :type currPDB: str
    """

    # Ensure the 'Results' directory exists
    result_dir = "Results"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Extract PDB ID and chain, format: PDB_ID \t Chain
    formatted_data = [f"{entry[:4]}\t{entry[4:]}" for entry in pdb_list]

    # Define output file path
    filename = f"{currPDB}_{chain}_Ensemble.txt"
    filepath = os.path.join(result_dir, filename)

    # Save the formatted data
    with open(filepath, "w") as f:
        f.write("PDB\tChain\n")  # Add header
        f.write("\n".join(formatted_data))

    LOGGER.info(f"Saved PDB ensemble list to {filepath}")


def getModesEnsemble(currPDB, eachChain, mode="auto", length=None, rmsd=None, Z=None, ref=None):
    """
    Retrieves and analyzes protein ensembles (similar or diverse) using Dali.

    :param currPDB: PDB ID or local PDB file.
    :type currPDB: str

    :param eachChain: Chain identifier of the target protein.
    :type eachChain: str

    :param mode: Selection method - "similar", "diverse", or "auto" (default).
                 "auto" decides based on protein size (550 residues as threshold).
    :type mode: str, default is "auto"

    :param length: Minimum fraction of sequence alignment length (0 to 1).
                   Defaults: 0.8 for "similar", 0.95 for "diverse".
    :type length: float, optional

    :param rmsd: Maximum allowed RMSD for filtering results.
                 Defaults: 2.0 for "similar", 1.0 for "diverse".
    :type rmsd: float, optional

    :param Z: Minimum Z-score threshold for structural similarity.
              Default is 10 for both modes.
    :type Z: float, optional

    :return: Tuple containing eigenvalues, eigenvectors, PDB IDs, and GNM results.
    :rtype: tuple
    """

    def getProteinLength(pdb_id, chain):
        """Returns the number of residues in a given PDB chain by counting alpha carbon (CA) atoms."""
        structure = parsePDB(pdb_id)
        chain_atoms = structure.select(f"chain {chain}")

        if chain_atoms is None:
            raise ValueError(f"Chain {chain} not found in {pdb_id}")

        ca_atoms = chain_atoms.select("name CA")
        if ca_atoms is None:
            raise ValueError(f"No alpha carbon (CA) atoms found in chain {chain} of {pdb_id}")

        return len(set(ca_atoms.getResnums()))

    # Determine protein length
    try:
        seq_residues = getProteinLength(currPDB, eachChain)
        LOGGER.warn(f"Residue length of query protein ({currPDB} chain {eachChain}): {seq_residues} residues.")
    except Exception as e:
        LOGGER.warn(f"Warning: Could not determine sequence length for {currPDB}. Defaulting to 'auto' mode. {str(e)}")
        seq_residues = None  # Proceed without length-based selection

    # Handling "auto" mode
    if mode == "auto":
        if seq_residues and 500 <= seq_residues <= 600:
            LOGGER.warn("Query protein is around 500-600 residues. Consider manually selecting 'similar' or 'diverse' ensemble using 'mode'.")
        mode = "similar" if seq_residues and seq_residues < 550 else "diverse"

    # Set default hyperparameters based on mode
    if mode == "similar":
        length = 0.8 if length is None else length
        rmsd = 2.0 if rmsd is None else rmsd
        Z = 10 if Z is None else Z
        LOGGER.info(f"Using 'similar' ensemble search for {currPDB} with length={length}, rmsd={rmsd}, Z={Z}.")
        dali_rec = searchDal(currPDB, eachChain)

    else:  # mode == "diverse"
        length = 0.95 if length is None else length
        rmsd = 1.0 if rmsd is None else rmsd
        Z = 10 if Z is None else Z
        LOGGER.info(f"Using 'diverse' ensemble search for {currPDB} with length={length}, rmsd={rmsd}, Z={Z}.")
        dali_rec = searchDali(currPDB, eachChain)

    # Fetch results if necessary
    while not dali_rec.isSuccess:
        dali_rec.fetch()

    # Apply Dali filters
    pdb_ids = dali_rec.filter(cutoff_len=length, cutoff_rmsd=rmsd, cutoff_Z=Z)
    LOGGER.info(f"# of {mode} structures found by Dali: {len(pdb_ids)}")

    # Retrieve mappings
    mappings = dali_rec.getMappings()

    # Build ensemble
    ags = parsePDB(pdb_ids, subset='ca')
    
    if ref is None:
        dali_ens = buildPDBEnsemble(ags, mapping=mappings, seqid=20)
        LOGGER.info("Building ensemble using default reference structure.")
    else:
        dali_ens = buildPDBEnsemble(ags, ref=ref)
        LOGGER.info("Building ensemble using provided reference structure.")
    
    savePDBList(pdb_ids, currPDB, eachChain)

    return pdb_ids


# ********************************** multiple chains **********************************************
def getTotalProteinLength(currPDB, chains):
    """
    Determines the total residue length of all given chains in a PDB.
    """
    structure = parsePDB(currPDB)
    total_length = 0

    for chain in chains:
        chain_atoms = structure.select(f"chain {chain}")
        if chain_atoms is None:
            raise ValueError(f"Chain {chain} not found in {currPDB}")

        ca_atoms = chain_atoms.select("name CA")
        if ca_atoms is None:
            raise ValueError(f"No alpha carbon (CA) atoms found in chain {chain} of {currPDB}")

        total_length += len(set(ca_atoms.getResnums()))

    return total_length

def getMergedMultiChainEnsemble(currPDB, chains, length=None, rmsd=None, Z=10):
    """
    Runs Dali for multi-chain proteins, determines mode based on total sequence length,
    merges results, and ensures only valid PDBs are included.
    """
    
    chain_list = list(chains)
    # Determine total sequence length and set mode
    try:
        total_length = getTotalProteinLength(currPDB, chains)
        LOGGER.info(f"Total sequence length of {currPDB} ({'+'.join(chains)}): {total_length} residues.")
    except Exception as e:
        LOGGER.warn(f"Could not determine sequence length for {currPDB}. Defaulting to 'auto' mode. {str(e)}")
        total_length = None

    # Define mode based on total sequence length
    mode = "diverse" if total_length and total_length > 550 else "similar"
    LOGGER.info(f"Using '{mode}' mode for {currPDB} based on total sequence length.")

    # Set default parameters based on mode
    if mode == "similar":
        length = 0.8 if length is None else length
        rmsd = 2.0 if rmsd is None else rmsd
    else:  # diverse
        length = 0.95 if length is None else length
        rmsd = 1.0 if rmsd is None else rmsd

    LOGGER.info(f"Using length={length}, rmsd={rmsd}, Z={Z} for {currPDB}.")

    # Store results for each chain
    chain_results = {}

    for chain in chains:
        LOGGER.info(f"Processing {currPDB} chain {chain}.")
        pdb_ids = getModesEnsemble(currPDB, chain, mode=mode, length=length, rmsd=rmsd, Z=Z)
        chain_results[chain] = set(pdb_ids)  # Convert to set for fast lookup

    # Merge chains while ensuring valid homologous structures
    merged_results = mergeChains(currPDB, chain_results, chain_list)

    # Save final merged results
    saveMultiChainResults(merged_results, currPDB)

    return merged_results

'''
def mergeChains(currPDB, chain_results, original_chains):
    """
    Merges multi-chain Dali results by identifying PDBs that contain
    homologous chains for all original chains, ensuring chains are distinct.
    """
    from collections import defaultdict

    # Build: pdb_id → {original_chain → set of chain IDs}
    pdb_map = defaultdict(lambda: defaultdict(set))
    for orig_chain, pdb_set in chain_results.items():
        for pdb in pdb_set:
            pdb_id, chain_id = pdb[:4], pdb[4:]
            pdb_map[pdb_id][orig_chain].add(chain_id)

    mergeIDs = []
    Chains = []

    for pdb_id, chains_by_orig in pdb_map.items():
        # Only consider PDBs that have homologs for *all* original chains
        if all(oc in chains_by_orig for oc in original_chains):
            # Create all pairwise combinations of different chain IDs from each original chain
            combos = []
            used_chain_ids = []
            valid = True

            for oc in original_chains:
                chain_ids = list(chains_by_orig[oc])
                if not chain_ids:
                    valid = False
                    break
                used_chain_ids.append(chain_ids)

            # Ensure distinct chain IDs (e.g., A ≠ B)
            if valid and len(set(c[0] for c in used_chain_ids)) == len(original_chains):
                merged_chain = ''.join(sorted(c[0] for c in used_chain_ids))
                mergeIDs.append(pdb_id)
                Chains.append(merged_chain)

    LOGGER.info(f"Merged {len(mergeIDs)} valid multi-chain homologous structures.")

    # Final results
    final_results = sorted(f"{pdb}{chain}" for pdb, chain in zip(mergeIDs, Chains))

    # Ensure reference structure appears first
    reference_entry = f"{currPDB}{''.join(sorted(original_chains))}"
    if reference_entry in final_results:
        final_results.remove(reference_entry)
        final_results.insert(0, reference_entry)

    return final_results
'''

# *****************************************************************************************************
# ******************************* integral two search methods ****************************************
from collections import defaultdict

def mergeChains(currPDB, chain_results, original_chains):
    """
    Merges multi-chain Dali results by identifying PDBs that contain
    homologous chains for all original chains, ensuring distinct chain usage.
    Avoids duplicates and excludes self-chain matches (e.g., AA or BB).
    """
    pdb_map = defaultdict(lambda: defaultdict(set))

    for orig_chain, pdb_set in chain_results.items():
        for pdb in pdb_set:
            pdb_id, chain_id = pdb[:4], pdb[4:]
            pdb_map[pdb_id][orig_chain].add(chain_id)

    seen = set()
    final_results = [currPDB]  # Always include the reference input (PDB ID or file name)

    for pdb_id, chains_by_orig in pdb_map.items():
        # Step 1: check if this pdb_id is in ALL input chains
        if not all(oc in chains_by_orig for oc in original_chains):
            continue

        # Step 2: try to form a combination of different chains across original chains
        valid = True
        combo = []
        used_chains = set()

        for oc in original_chains:
            candidates = chains_by_orig[oc]
            # Pick first unused chain ID
            chosen = None
            for c in candidates:
                if c not in used_chains:
                    chosen = c
                    break
            if chosen is None:
                valid = False
                break
            combo.append(chosen)
            used_chains.add(chosen)

        if not valid:
            continue

        # Step 3: create final string and avoid duplicate combos like AB vs BA
        chain_str = ''.join(sorted(combo))
        tag = f"{pdb_id}{chain_str}"
        if tag not in seen:
            seen.add(tag)
            final_results.append(tag)

    if len(final_results) > 1:
        return final_results[1:]
    else:
        return []


def getEnsembleWithModes(currPDB, chains, mode="auto", length=None, rmsd=None, Z=10):
    """
    Returns a list of homologous structure IDs using Dali (with local .pdb or PDB ID).
    Supports both monomer and multimer chains.

    If a .pdb file is used, it is inserted as the first element of the returned list.

    :param currPDB: str - PDB ID or local .pdb file
    :param chains: str - single or multiple chain IDs (e.g., "A" or "AB")
    :param mode: "similar", "diverse", or "auto"
    :param length: float - Dali alignment cutoff
    :param rmsd: float - Dali RMSD filter
    :param Z: float - Dali Z-score filter
    :return: list of structure IDs (strings)
    """

    is_file = currPDB.endswith(".pdb") and os.path.isfile(currPDB)
    chain_list = list(chains)

    # Determine mode if "auto"
    if mode == "auto":
        try:
            total_length = getTotalProteinLength(currPDB, chains)
            LOGGER.info(f"Total sequence length of {currPDB} ({'+'.join(chains)}): {total_length} residues.")
            mode = "similar" if total_length < 550 else "diverse"
        except Exception as e:
            LOGGER.warn(f"Could not determine sequence length for {currPDB}. Defaulting to 'diverse'. {str(e)}")
            mode = "diverse"

    # Set Dali params
    if mode == "similar":
        length = 0.8 if length is None else length
        rmsd = 2.0 if rmsd is None else rmsd
        search_func = searchDal
    else:
        length = 0.90 if length is None else length
        rmsd = 1.0 if rmsd is None else rmsd
        search_func = searchDali

    LOGGER.info(f"Using '{mode}' mode with length={length}, rmsd={rmsd}, Z={Z}")

    # Monomer
    if len(chain_list) == 1:
        chain = chain_list[0]
        dali_rec = search_func(currPDB, chain)
        while not dali_rec.isSuccess:
            dali_rec.fetch()

        pdb_ids = dali_rec.filter(cutoff_len=length, cutoff_rmsd=rmsd, cutoff_Z=Z)
        if is_file:
            pdb_ids.insert(0, currPDB)
        savePDBList(pdb_ids, currPDB, chain)

        return pdb_ids

    # Multimer
    chain_results = {}
    for chain in chain_list:
        LOGGER.info(f"Processing {currPDB} chain {chain}")
        dali_rec = search_func(currPDB, chain)
        while not dali_rec.isSuccess:
            dali_rec.fetch()
        chain_results[chain] = set(dali_rec.filter(cutoff_len=length, cutoff_rmsd=rmsd, cutoff_Z=Z))

    merged_results = mergeChains(currPDB, chain_results, chain_list)
    if not merged_results:
        LOGGER.warn(f"No valid merged multi-chain homologs found for {currPDB}. Returning only reference.")
        return [currPDB] if is_file else []

    if not is_file:
        ref = currPDB + chains
        merged_results.insert(0, ref)
        merged_results = [item for item in merged_results if item != ref]
        merged_results.insert(0, ref)
    else:
        merged_results.insert(0, currPDB)
    
    saveMultiChainResults(merged_results, currPDB)
    return merged_results



def saveMultiChainResults(merged_results, currPDB):
    """
    Saves the final merged homologous PDB structures to the 'Results' folder.
    """

    result_dir = "Results"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    filename = f"{currPDB}_MultiChainEnsemble.txt"
    filepath = os.path.join(result_dir, filename)

    with open(filepath, "w") as f:
        f.write("PDB\tChains\n")  # Header
        for entry in merged_results:
            f.write(f"{entry[:4]}\t{entry[4:]}\n")

    LOGGER.info(f"Saved multi-chain ensemble list to {filepath}")
    
    
   
# *************************************************************************************************
# **************************************** GNM mode ***********************************************
# *************************************************************************************************
'''
def motionFromEnsemble(input_data, ref=None, from_file=False):
    """
    Computes GNM (Gaussian Network Model) from an ensemble list or a text file.

    :param input_data: Either a list of PDB chain identifiers or a path to a text file containing them.
    :type input_data: list of str or str (file path)

    :param ref: Reference structure for ensemble alignment. Default is the first structure in your list.
    :type ref: Atomic object (optional)

    :param from_file: Whether input_data is a file path. Default is False.
    :type from_file: bool

    :return: GNM object, average eigenvalues, average eigenvectors
    :rtype: tuple (GNM, np.array, np.array)
    """

    # Read data from file if from_file=True
    if from_file:
        try:
            with open(input_data, "r") as f:
                lines = f.readlines()[1:]  # Skip header
                pdb_chains = [line.strip().replace("\t", "") for line in lines]
        except FileNotFoundError:
            LOGGER.error(f"File {input_data} not found.")
            return None, None, None
    else:
        pdb_chains = input_data  # Directly use provided list

    LOGGER.info(f"Processing {len(pdb_chains)} PDB entries.")
    
    # Extract PDB IDs and Chains separately
    if len(pdb_chains[0]) == 5:
        ags = parsePDB(pdb_chains, subset="ca")
    else:
        mergeIDs = [entry[:4] for entry in pdb_chains]  # First 4 chars = PDB ID
        Chains = [entry[4:] for entry in pdb_chains]  # Remaining chars = Chains
        
        ags = parsePDB(mergeIDs, subset="ca", chain=Chains[0])

    # Build ensemble with or without ref
    if ref:
        dali_ens = buildPDBEnsemble(ags, ref=ref)
    else:
        dali_ens = buildPDBEnsemble(ags)

    # Compute GNM
    gnms = calcEnsembleENMs(dali_ens, model="GNM", trim="reduce", n_modes=None)

    # Extract eigenvalues & eigenvectors
    eigVals = gnms.getEigvals()
    averageEigVals = sort(np.mean(eigVals, axis=0))
    eigVects = gnms.getEigvecs()
    averageEigVecs = np.mean(eigVects, axis=0)

    LOGGER.info(f"GNM computation completed with {len(eigVals)} eigenvalues.")

    return gnms, averageEigVals, averageEigVecs
'''

def motionFromEnsemble_AF(input_data, ref=None, from_file=False):
    """
    Computes GNM (Gaussian Network Model) from an ensemble list or a text file.

    Args:
        input_data (list[str] or str): List of PDB+Chain identifiers (e.g. '1tklA' or '1tklAB')
                                       or file path to such a list.
        ref (Atomic): Reference structure for alignment (optional).
        from_file (bool): If True, reads list from file.

    Returns:
        tuple: (GNM object, average eigenvalues, average eigenvectors)
    """
    import os

    if from_file:
        try:
            with open(input_data, "r") as f:
                lines = f.readlines()[1:]  # Skip header
                pdb_chains = [line.strip().replace("\t", "") for line in lines]
        except FileNotFoundError:
            LOGGER.error(f"File {input_data} not found.")
            return None, None, None
    else:
        pdb_chains = input_data

    if not pdb_chains:
        LOGGER.error("No PDB entries provided.")
        return None, None, None

    if len(pdb_chains) == 1:
        LOGGER.warning("Only one structure provided; duplicating it for ensemble.")
        pdb_chains.append(pdb_chains[0])

    LOGGER.info(f"Processing {len(pdb_chains)} PDB entries.")

    all_atoms = []

    for entry in pdb_chains:
        if os.path.isfile(entry):
            # It's a custom .pdb file (e.g., "CPOX_AF.pdb")
            ag = parsePDB(entry, subset="ca")
            all_atoms.append(ag)
        else:
            # It's a PDB ID with one or more chains (e.g., "1tklA", "1tklAB")
            pdb_id = entry[:4]
            chain_ids = entry[4:]

            if not chain_ids:
                ag = parsePDB(pdb_id, subset="ca")
                all_atoms.append(ag)
            else:
                ag = parsePDB(pdb_id, chain=chain_ids, subset="ca")
                all_atoms.append(ag)

    if ref:
        ensemble = buildPDBEnsemble(all_atoms, ref=ref)
    else:
        ensemble = buildPDBEnsemble(all_atoms)

    gnms = calcEnsembleENMs(ensemble, model="GNM", trim="reduce", n_modes=None)


    eigVals = gnms.getEigvals()
    averageEigVals = sort(np.mean(eigVals, axis=0))
    eigVects = gnms.getEigvecs()
    averageEigVecs = np.mean(eigVects, axis=0)


    return gnms, averageEigVals, averageEigVecs

def motionFromEnsemble_PDB(input_data, ref=None, from_file=False):
    """
    Computes GNM (Gaussian Network Model) from an ensemble list or a text file.

    :param input_data: Either a list of PDB chain identifiers or a path to a text file containing them.
    :type input_data: list of str or str (file path)

    :param ref: Reference structure for ensemble alignment. Default is the first structure in your list.
    :type ref: Atomic object (optional)

    :param from_file: Whether input_data is a file path. Default is False.
    :type from_file: bool

    :return: GNM object, average eigenvalues, average eigenvectors
    :rtype: tuple (GNM, np.array, np.array)
    """

    # Read data from file if from_file=True
    if from_file:
        try:
            with open(input_data, "r") as f:
                lines = f.readlines()[1:]  # Skip header
                pdb_chains = [line.strip().replace("\t", "") for line in lines]
        except FileNotFoundError:
            LOGGER.error(f"File {input_data} not found.")
            return None, None, None
    else:
        pdb_chains = input_data  # Directly use provided list

    LOGGER.info(f"Processing {len(pdb_chains)} PDB entries.")
    
    # Extract PDB IDs and Chains separately
    if len(pdb_chains[0]) == 5:
        ags = parsePDB(pdb_chains, subset="ca")
    else:
        mergeIDs = [entry[:4] for entry in pdb_chains]  # First 4 chars = PDB ID
        Chains = [entry[4:] for entry in pdb_chains]  # Remaining chars = Chains
        
        ags = parsePDB(mergeIDs, subset="ca", chain=Chains[0])

    # Build ensemble with or without ref
    if ref:
        dali_ens = buildPDBEnsemble(ags, ref=ref)
    else:
        dali_ens = buildPDBEnsemble(ags)

    # Compute GNM
    gnms = calcEnsembleENMs(dali_ens, model="GNM", trim="reduce", n_modes=None)

    # Extract eigenvalues & eigenvectors
    eigVals = gnms.getEigvals()
    averageEigVals = sort(np.mean(eigVals, axis=0))
    eigVects = gnms.getEigvecs()
    averageEigVecs = np.mean(eigVects, axis=0)

    LOGGER.info(f"GNM computation completed with {len(eigVals)} eigenvalues.")

    return gnms, averageEigVals, averageEigVecs


def motionFromEnsemble(input_data, ref=None, from_file=False):
    if '.pdb' in input_data[0]:
        gnms, avg_eigvals, avg_eigvecs = motionFromEnsemble_AF(input_data, ref=ref)
    else:
        gnms, avg_eigvals, avg_eigvecs = motionFromEnsemble_PDB(input_data, ref=ref)
    return gnms, avg_eigvals, avg_eigvecs
# ******************************************************************************************


def computeModeContribution(eigenVals, mode):
    """
    Computes the relative contribution of the first `mode` modes to the total 
    contribution in a Gaussian Network Model (GNM).

    :param eigenVals: Array of eigenvalues from GNM.
    :type eigenVals: np.array

    :param mode: Number of modes to consider.
    :type mode: int

    :return: The contribution ratio of the first `mode` modes.
    :rtype: float
    """
    total_contribution = np.sqrt(sum(1 / eigenVals))  # Total contribution from all modes
    selected_modes_contribution = np.sqrt(sum(1 / eigenVals[:mode]))  # Contribution from selected modes
    return selected_modes_contribution / total_contribution  # Normalized contribution


def findModesForThreshold(eigenVals, threshold):
    """
    Finds the minimum number of modes required to reach a given contribution threshold.

    :param eigenVals: Array of eigenvalues from GNM.
    :type eigenVals: np.array

    :param threshold: Desired contribution level (e.g., 0.9 for 90% contribution).
    :type threshold: float

    :return: Number of modes required to reach the threshold.
    :rtype: int
    """
    contribution = 0
    mode = 0
    while contribution <= threshold:
        mode += 1
        contribution = computeModeContribution(eigenVals, mode)  # Reuse renamed function
    return mode


# *************************************************************************************************
# ***************************************** get hinges ********************************************
# *************************************************************************************************
def flattenArray(twoDArray):
    """
    Flattens a 2D list into a 1D list.

    :param twoDArray: A 2D list or array.
    :type twoDArray: list of list

    :return: Flattened 1D list.
    :rtype: list
    """
    return [item for sublist in twoDArray for item in sublist]


def mergeOverlappingRegions(regions):
    """
    Merges overlapping regions based on numerical proximity.

    :param regions: A list of lists, where each inner list contains indices of a region.
    :type regions: list of list

    :return: A list of merged regions with no overlaps.
    :rtype: list of list
    """
    sorted_regions = sorted(regions, key=lambda x: x[0])  # Sort regions by the first element
    merged_regions = [sorted_regions[0]]

    for current in sorted_regions[1:]:
        previous = merged_regions[-1]
        if current[0] <= previous[-1]:  # Overlapping regions
            merged_regions[-1] = list(range(min(previous[0], current[0]), max(previous[-1], current[-1]) + 1))
        else:
            merged_regions.append(current)
    
    return merged_regions


def detectSingleModeHinges(eigenVector, threshold=20):
    """
    Identifies hinge sites in a single mode of an eigenvector and separates 
    primary and minor hinge sites.

    :param eigenVector: A 1D array representing a mode's eigenvector values.
    :type eigenVector: np.array

    :param threshold: The bandwidth threshold for identifying hinge sites.
    :type threshold: float

    :return: Two lists - Primary hinge sites and minor hinge sites.
    :rtype: tuple (list of lists, list of lists)
    """
    band = (-np.sqrt(1 / len(eigenVector)) / threshold, np.sqrt(1 / len(eigenVector)) / threshold)

    # Identify crossover points (sign changes)
    crossovers = [i for i in range(1, len(eigenVector)) if 
                  (eigenVector[i-1] < 0 and eigenVector[i] > 0) or 
                  (eigenVector[i-1] > 0 and eigenVector[i] < 0)]

    regions = []
    for i in crossovers:
        region = [i-1, i]  # Start with adjacent indices
        
        if (region[0] < band[0] and region[1] > band[1]) or (region[0] < band[1] and region[1] > band[0]):
            regions.append(region)
        else:
            # Expand backwards
            j = i - 2
            while j >= 0 and band[0] <= eigenVector[j] <= band[1]:
                region.insert(0, j)
                j -= 1
            
            # Expand forwards
            j = i + 1
            while j < len(eigenVector) and band[0] < eigenVector[j] < band[1]:
                region.append(j)
                j += 1

            regions.append(region)

    # Merge overlapping regions
    merged_regions = mergeOverlappingRegions(regions)

    final_regions = []
    minor_regions = []
    
    for region in merged_regions:
        if len(region) >= 5:
            pos = sorted([idx for idx in region if eigenVector[idx] > 0], key=lambda x: eigenVector[x])
            neg = sorted([idx for idx in region if eigenVector[idx] < 0], key=lambda x: eigenVector[x], reverse=True)
            n = int(len(region) / 4) + 1
            primary_region = pos[:n] + neg[:n]
            minor_region = [idx for idx in region if idx not in primary_region]  # Store removed indices
            
        elif 3 <= len(region) <= 4:
            if eigenVector[region[0] - 1] * eigenVector[region[-1] + 1] > 0:
                continue  # Skip this region if adjacent values outside the band have the same sign
            elif len(region) == 3:
                absolute_values = [abs(eigenVector[i]) for i in region]
                smallest_two_indices = np.argsort(absolute_values)[:2]
                primary_region = [region[i] for i in smallest_two_indices]
                minor_region = [idx for idx in region if idx not in primary_region]
            else:
                primary_region = region
                minor_region = []
        else:
            primary_region = region
            minor_region = []

        if primary_region not in final_regions:
            final_regions.append(primary_region)

        if minor_region and minor_region not in minor_regions:
            minor_regions.append(minor_region)

    return final_regions, minor_regions


def identifyHingeSites(eigenVectors, eigenValues, numModes=None, threshold=15):
    """
    Identifies hinge sites based on eigenvectors and a specified number of modes.
    Separates primary hinges and minor hinges.

    :param eigenVectors: A 2D array of eigenvectors (each column represents a mode).
    :type eigenVectors: np.array

    :param eigenValues: A 1D array of eigenvalues.
    :type eigenValues: np.array

    :param numModes: Number of modes to consider for hinge detection. If None, auto-selects min(33% contribution, 3).
    :type numModes: int or None

    :param threshold: Bandwidth threshold for hinge detection.
    :type threshold: float

    :return: Two lists - Primary hinge site indices and minor hinge site indices.
    :rtype: tuple (list, list)
    """
    # Auto-select numModes if not provided
    if numModes is None:
        contribution_threshold = 0.33  # 33% contribution
        estimated_modes = findModesForThreshold(eigenValues, contribution_threshold)
        numModes = min(estimated_modes, 3)  # Use min(33% contribution, 3)
        LOGGER.info(f"Auto-selected {numModes} modes based on contribution threshold.")

    primary_hinges = set()
    minor_hinges = set()

    for i in range(numModes):
        primary_regions, minor_regions = detectSingleModeHinges(eigenVectors[:, i], threshold=threshold)
        primary_hinges.update(flattenArray(primary_regions))
        minor_hinges.update(flattenArray(minor_regions))

    return sorted(primary_hinges), sorted(minor_hinges)


def mapHingesToResidues(pdb_file, hinge_indices, chain=None, reference_selection=None):
    """
    Maps hinge site indices to actual residue names, numbers, and chain identifiers from a PDB file.

    :param pdb_file: Path to the PDB file.
    :type pdb_file: str

    :param hinge_indices: List of hinge indices (0-based).
    :type hinge_indices: list

    :param chain: Specific chain(s) to extract hinge sites from. If None, all chains are considered.
    :type chain: str or list or None

    :param reference_selection: ProDy AtomGroup object representing the selected reference structure (e.g., only TM domain).
    :type reference_selection: prody.atomic.atomgroup.AtomGroup or None

    :return: List of hinge residues in 'ResidueName-ResidueNumber(Chain)' format.
    :rtype: list
    """
    # Parse full PDB structure if no reference selection is provided
    if reference_selection is None:
        structure = parsePDB(pdb_file, subset='ca', chain=chain)
    else:
        structure = reference_selection  # Use the provided AtomGroup selection

    resnums = structure.getResnums()  # Get actual residue numbers
    resnames = structure.getResnames()  # Get residue names
    chains = structure.getChids()  # Get chain identifiers

    # Ensure hinge indices are within valid range of selected reference
    hinge_residues = [
        f"{resnames[i]}{resnums[i]}({chains[i]})"
        for i in hinge_indices if i < len(resnums)
    ]
    
    return hinge_residues
    

def saveHingeResidues(pdb_file, chain, eigenVectors, eigenValues, reference_selection=None, threshold=15, output_dir="Results"):
    """
    Identifies and saves hinge residues (both major and minor) from GNM analysis.

    :param pdb_file: Path to the PDB file.
    :type pdb_file: str

    :param chain: Chain identifier(s) used in analysis.
    :type chain: str or list

    :param eigenVectors: A 2D array of eigenvectors (each column represents a mode).
    :type eigenVectors: np.array

    :param eigenValues: A 1D array of eigenvalues.
    :type eigenValues: np.array

    :param reference_selection: ProDy AtomGroup object representing the selected reference structure.
    :type reference_selection: prody.atomic.atomgroup.AtomGroup or None

    :param threshold: Bandwidth threshold for hinge detection.
    :type threshold: float

    :param output_dir: Directory to save the result file.
    :type output_dir: str
    """
    # Ensure the Results directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        LOGGER.info(f"Created directory: {output_dir}")

    # Identify hinge sites (major and minor)
    major_hinges, minor_hinges = identifyHingeSites(eigenVectors, eigenValues, threshold=threshold)

    # Map hinge sites to actual residue names and numbers
    major_residues = mapHingesToResidues(pdb_file, major_hinges, chain=chain, reference_selection=reference_selection)
    minor_residues = mapHingesToResidues(pdb_file, minor_hinges, chain=chain, reference_selection=reference_selection)

    # Prepare output file path
    pdb_id = os.path.basename(pdb_file).split(".")[0]  # Extract PDB ID from file name
    output_file = os.path.join(output_dir, f"{pdb_id}_{chain}_HingeResidues.txt")

    # Write to file
    with open(output_file, "w") as f:
        f.write(f"Major: {', '.join(major_residues)}\n")
        f.write(f"Minor: {', '.join(minor_residues)}\n")

    LOGGER.info(f"Hinge residues saved to {output_file}")


def getGlobalHinges(currPDB, chains, gnms, ref=None, save_file=True, numModes=None, threshold=15):
    eigVals = gnms.getEigvals()
    averageEigVals = sort(np.mean(eigVals, axis=0))
    eigVects = gnms.getEigvecs()
    averageEigVecs = np.mean(eigVects, axis=0)
    LOGGER.info(f"GNM computation completed with {len(eigVals)} eigenvalues.")
    
    primary, minor = identifyHingeSites(averageEigVecs, averageEigVals, numModes=numModes, threshold=threshold)
    
    hinge_residues = {'primary': mapHingesToResidues
                      (currPDB, primary, chain=chains, reference_selection=ref), 
                      'minor': mapHingesToResidues(currPDB, minor, chain=chains, reference_selection=ref)}

    if save_file:
        LOGGER.info("Saved the hinge residues")
        saveHingeResidues(currPDB, chains, averageEigVecs, averageEigVals, reference_selection=ref)
    
    return hinge_residues

# ***************************************** get binding sites ******************************************
import gzip

### Step 1: Read `.pdb.gz` file and extract atomic data ###
def parse_pdb_gz(pdb_id, pdb_dir="./"):
    """
    Parses a `.pdb.gz` file and extracts protein and ligand atomic coordinates.

    :param pdb_id: The PDB ID of the structure.
    :type pdb_id: str

    :param pdb_dir: Directory containing the `.pdb.gz` files.
    :type pdb_dir: str

    :return: Dictionary with protein and ligand atom coordinates.
    :rtype: dict
    """
    pdb_path = os.path.join(pdb_dir, f"{pdb_id}.pdb.gz")

    if not os.path.exists(pdb_path):
        LOGGER.warning(f"PDB file {pdb_path} not found.")
        return None

    protein_atoms = {}  # {chain: [(residue_name, residue_id, atom_name, x, y, z), ...]}
    ligand_atoms = {}   # {ligand_id: [(chain, residue_id, atom_name, x, y, z), ...]}

    # Open and read .pdb.gz file
    with gzip.open(pdb_path, "rt") as f:
        for line in f:
            if line.startswith("ATOM"):  # Protein atoms
                atom_name = line[12:16].strip()
                residue_name = line[17:20].strip()
                chain = line[21].strip()
                residue_id = line[22:26].strip()
                x, y, z = map(float, [line[30:38], line[38:46], line[46:54]])

                if chain not in protein_atoms:
                    protein_atoms[chain] = []
                protein_atoms[chain].append((residue_name, residue_id, atom_name, x, y, z))

            elif line.startswith("HETATM"):  # Ligand atoms (excluding water)
                residue_name = line[17:20].strip()
                if residue_name == "HOH":
                    continue  # Skip water molecules

                atom_name = line[12:16].strip()
                chain = line[21].strip()
                residue_id = line[22:26].strip()
                x, y, z = map(float, [line[30:38], line[38:46], line[46:54]])

                ligand_id = f"{residue_name}_{residue_id}"
                if ligand_id not in ligand_atoms:
                    ligand_atoms[ligand_id] = []
                ligand_atoms[ligand_id].append((chain, residue_id, atom_name, x, y, z))

    LOGGER.info(f"Parsed {pdb_id}.pdb.gz: Found {len(protein_atoms)} protein chains and {len(ligand_atoms)} ligands.")
    return {"protein": protein_atoms, "ligand": ligand_atoms}
    

'''
def find_binding_residues(protein_atoms, ligand_atoms, threshold=5.0):
    """
    Identifies protein residues within a distance threshold from ligand atoms.

    :param protein_atoms: ProDy AtomGroup containing protein atoms.
    :param ligand_atoms: ProDy AtomGroup containing ligand atoms.
    :param threshold: Distance cutoff in Å (default = 5.0).
    :return: Dictionary mapping chains to lists of binding residues.
    """
    binding_sites = {}
    
    for chain in protein_atoms.getHierView():  # Iterate over chains
        chain_atoms = protein_atoms.select(f'chain {chain}')
        if chain_atoms is None:
            continue

        chain_residues = set()
        for res in chain_atoms.getHierView():  # Iterate over residues
            res_atoms = chain_atoms.select(f'resnum {res.getResnum()}')
            if res_atoms is None:
                continue

            # Compute pairwise distances between residue and ligand atoms
            distances = np.linalg.norm(res_atoms.getCoords()[:, None, :] - ligand_atoms.getCoords(), axis=-1)
            min_distance = min(distances)

            if min_distance <= threshold:
                res_name = res.getResname()
                res_num = res.getResnum()
                chain_residues.add(f"{res_name}{res_num}")

        if chain_residues:
            binding_sites[chain] = sorted(chain_residues)
    
    return binding_sites
'''

def find_binding_residues(protein_atoms, ligand_atoms, threshold=5.0):
    """
    Identifies protein residues within a distance threshold from ligand atoms.
    
    :param protein_atoms: Dictionary containing protein atom coordinates {chain: [(resname, resnum, atom, x, y, z), ...]}.
    :param ligand_atoms: Dictionary containing ligand atom coordinates {ligand: [(chain, resnum, atom, x, y, z), ...]}.
    :param threshold: Distance cutoff in Å (default = 5.0).
    :return: Dictionary mapping ligands to chains and binding residues {ligand: {chain: [residue_list]}}.
    """
    binding_sites = {}
    
    for ligand, ligand_atoms_list in ligand_atoms.items():
        binding_sites[ligand] = {}
        
        for ligand_atom in ligand_atoms_list:
            lig_chain, lig_resnum, lig_atom, lx, ly, lz = ligand_atom
            
            for chain, protein_atom_list in protein_atoms.items():
                for protein_atom in protein_atom_list:
                    resname, resnum, atom, px, py, pz = protein_atom
                    
                    # Compute Euclidean distance
                    distance = sqrt((px - lx) ** 2 + (py - ly) ** 2 + (pz - lz) ** 2)
                    
                    if distance <= threshold:
                        if chain not in binding_sites[ligand]:
                            binding_sites[ligand][chain] = set()
                        binding_sites[ligand][chain].add(f"{resname}{resnum}")
    
    # Convert sets to sorted lists
    for ligand in binding_sites:
        for chain in binding_sites[ligand]:
            binding_sites[ligand][chain] = sorted(binding_sites[ligand][chain])
    
    return binding_sites

def parse_pdb_AF(pdb_path):
    """
    Parses a `.pdb` or `.txt` file and extracts protein and ligand atomic coordinates.

    :param pdb_path: Path to the PDB file (.pdb or .txt).
    :type pdb_path: str

    :return: Dictionary with protein and ligand atom coordinates.
    :rtype: dict
    """
    if not os.path.exists(pdb_path):
        LOGGER.warning(f"PDB file {pdb_path} not found.")
        return None

    protein_atoms = {}  # {chain: [(residue_name, residue_id, atom_name, x, y, z), ...]}
    ligand_atoms = {}   # {ligand_id: [(chain, residue_id, atom_name, x, y, z), ...]}

    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("ATOM"):
                atom_name = line[12:16].strip()
                residue_name = line[17:20].strip()
                chain = line[21].strip()
                residue_id = line[22:26].strip()
                x, y, z = map(float, [line[30:38], line[38:46], line[46:54]])

                protein_atoms.setdefault(chain, []).append(
                    (residue_name, residue_id, atom_name, x, y, z)
                )

            elif line.startswith("HETATM"):
                residue_name = line[17:20].strip()
                if residue_name == "HOH":
                    continue  # Skip water

                atom_name = line[12:16].strip()
                chain = line[21].strip()
                residue_id = line[22:26].strip()
                x, y, z = map(float, [line[30:38], line[38:46], line[46:54]])

                ligand_id = f"{residue_name}_{residue_id}"
                ligand_atoms.setdefault(ligand_id, []).append(
                    (chain, residue_id, atom_name, x, y, z)
                )

    LOGGER.info(f"Parsed {os.path.basename(pdb_path)}: Found {len(protein_atoms)} protein chains and {len(ligand_atoms)} ligands.")
    return {"protein": protein_atoms, "ligand": ligand_atoms}


def process_pdb_ligand_binding(pdb_list, threshold=5.0, pdb_dir="./", output_dir="Results"):
    """
    Identifies ligand binding sites for multiple PDB structures from `.pdb.gz` files.

    :param pdb_list: List of PDB structure identifiers.
    :type pdb_list: list

    :param threshold: Distance cutoff for ligand binding (default = 5.0 Å).
    :type threshold: float

    :param pdb_dir: Directory containing the `.pdb.gz` files.
    :type pdb_dir: str

    :param output_dir: Directory to save results.
    :type output_dir: str

    :return: Dictionary mapping PDB structures to their ligand binding sites.
    :rtype: dict
    """
    # Ensure Results folder exists
    os.makedirs(output_dir, exist_ok=True)

    binding_results = {}

    for pdb_entry in pdb_list:
        pdb_id = pdb_entry[:4]  # Try first 4 letters as PDB ID
        tried_paths = []
    
        # Try exact name (e.g., "CPOX_AF.pdb")
        pdb_path_1 = os.path.join(pdb_dir, f"{pdb_entry}")
    
        # Try gzipped PDB format (e.g., "1tkl.pdb.gz")
        pdb_path_2 = os.path.join(pdb_dir, f"{pdb_id}.pdb.gz")
        
        # Try them in order
        if os.path.exists(pdb_path_1):
            structure_file = pdb_path_1
            parsed_data = parse_pdb_AF(structure_file)
        elif os.path.exists(pdb_path_2):
            structure_file = pdb_id
            parsed_data = parse_pdb_gz(structure_file)
        else:
            structure_file = None
        
        if structure_file is None:
            LOGGER.warn(f"Skipping {pdb_entry}: no .pdb or .pdb.gz file found in {pdb_dir}")
            continue

        # Find ligand binding sites
        protein_atoms = parsed_data["protein"]
        ligand_atoms = parsed_data["ligand"]
        binding_sites = find_binding_residues(protein_atoms, ligand_atoms, threshold)

        if binding_sites:
            binding_results[pdb_id] = binding_sites

    # Save results to a TSV file
    output_file = os.path.join(output_dir, "LigandBindingSites.tsv")
    with open(output_file, "w") as f:
        f.write("PDB\tLigand\tChain\tBinding Residues\n")
        for pdb_id, ligands in binding_results.items():
            for ligand, chains in ligands.items():
                for chain, residues in chains.items():
                    f.write(f"{pdb_id}\t{ligand}\t{chain}\t{', '.join(residues)}\n")

    LOGGER.info(f"Ligand binding site results saved to {output_file}")
    return binding_results

def filter_ligands(binding_sites, approved_ligands):
    """
    Filters the binding_sites dictionary based on two criteria:
    1. The ligand must be in the approved ligand list.
    2. The ligand must have at least one chain with 15 or more binding sites.

    :param binding_sites: Dictionary containing binding site information.
    :param approved_ligands: Set of approved ligand 3-letter codes.
    :return: Filtered binding_sites dictionary.
    """
    filtered_binding_sites = {}

    for pdb_id, ligands in binding_sites.items():
        filtered_ligands = {}

        for ligand_name, binding_data in ligands.items():
            ligand_code = ligand_name.split("_")[0]  # Extract 3-letter ligand code
            if ligand_code not in approved_ligands:
                continue  # Skip ligands not in the approved list

            # Find the max number of binding sites across chains
            max_binding_sites = max(len(sites) for sites in binding_data.values())

            
            if max_binding_sites >= 15:  # Only keep ligands with sufficient binding sites
                filtered_ligands[ligand_name] = binding_data

        if filtered_ligands:  # Only keep entries with valid ligands
            filtered_binding_sites[pdb_id] = filtered_ligands

    return filtered_binding_sites

# *************************************************************************************************
# ***************************************** binding site ******************************************
# *************************************************************************************************
def extract_pdb_ligand_iupac_gz(pdb_id):
    """
    Extracts ligands (3-letter codes) and their IUPAC names from a `.pdb.gz` file.
    
    :param pdb_id: PDB ID to process.
    :return: Dictionary {Ligand Code: IUPAC Name}
    """
    pdb_file = os.path.join('./', f"{pdb_id}.pdb.gz")
    if not os.path.exists(pdb_file):
        LOGGER.info(f"PDB file {pdb_id}.pdb.gz not found!")
        return {}

    ligand_data = {}  # {Ligand Code: IUPAC Name}
    current_ligand = None  # Track the ligand being processed
    current_iupac = []  # Store multi-line IUPAC name parts

    with gzip.open(pdb_file, "rt") as f:
        for line in f:
            # Match HETNAM entries, handling multi-line ligand names
            match = re.match(r"HETNAM\s+(\d*)\s*(\S+)\s+(.+)", line)
            if match:
                num, ligand_code, iupac_part = match.groups()

                # If `num` is empty, it's a new ligand
                if num.isdigit():
                    if current_ligand and current_iupac:
                        ligand_data[current_ligand] = " ".join(current_iupac)  # Save previous ligand
                    current_iupac.append(iupac_part.strip())  # Append new part to the ongoing ligand
                else:
                    # New ligand starts, save the previous one
                    if current_ligand and current_iupac:
                        ligand_data[current_ligand] = " ".join(current_iupac)
                    
                    current_ligand = ligand_code
                    current_iupac = [iupac_part.strip()]  # Start new IUPAC name

        # Save the last ligand processed
        if current_ligand and current_iupac:
            ligand_data[current_ligand] = " ".join(current_iupac)

    return ligand_data

def extract_ligands_for_pdb_list(pdb_list, output_file="Results/All_Ligands.txt"):
    """
    Processes a list of PDB IDs, extracts their ligands (3-letter codes) and IUPAC names, 
    and saves the results in a tab-separated text file.

    :param pdb_list: List of PDB IDs to process.
    :param output_file: Path to the output text file.
    """
    pdb_list = [item[:4] for item in pdb_list]
    with open(output_file, "w") as f:
        for pdb_id in pdb_list:
            # Extract ligand data from the .pdb.gz file
            ligand_data = extract_pdb_ligand_iupac_gz(pdb_id)
            if not ligand_data:
                continue

            # Write results: Each ligand on a separate line
            for ligand, iupac in ligand_data.items():
                f.write(f"{pdb_id}\t{ligand}\t{iupac}\n")

    LOGGER.info(f"\n Ligand IUPAC names saved to {output_file}")

def load_approved_ligands(file_path):
    """
    Loads the list of FDA-approved ligands from the file.
    
    :param file_path: Path to the filtered ligand file.
    :return: Set of ligand 3-letter codes.
    """
    approved_ligands = set()
    with open(file_path, "r") as f:
        next(f)  # Skip header
        for line in f:
            fields = line.strip().split("\t")
            if len(fields) < 4:
                continue  # Skip incomplete lines
            ligands = fields[3].split(", ")  # Extract ligands
            approved_ligands.update(ligands)  # Add ligands to the set
    return approved_ligands

def filter_ligands(binding_sites, approved_ligands):
    """
    Filters the binding_sites dictionary based on two criteria:
    1. The ligand must be in the approved ligand list.
    2. The ligand must have at least one chain with 15 or more binding sites.

    :param binding_sites: Dictionary containing binding site information.
    :param approved_ligands: Set of approved ligand 3-letter codes.
    :return: Filtered binding_sites dictionary.
    """
    filtered_binding_sites = {}

    for pdb_id, ligands in binding_sites.items():
        filtered_ligands = {}

        for ligand_name, binding_data in ligands.items():
            ligand_code = ligand_name.split("_")[0]  # Extract 3-letter ligand code
            if ligand_code not in approved_ligands:
                continue  # Skip ligands not in the approved list

            # Find the max number of binding sites across chains
            max_binding_sites = max(len(sites) for sites in binding_data.values())

            if max_binding_sites >= 15:  # Only keep ligands with sufficient binding sites
                filtered_ligands[ligand_name] = binding_data

        if filtered_ligands:  # Only keep entries with valid ligands
            filtered_binding_sites[pdb_id] = filtered_ligands

    return filtered_binding_sites

# Mapping of 3-letter amino acid codes to 1-letter codes
AA_DICT = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
    'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

def three_to_one(res):
    """Converts 3-letter amino acid code to 1-letter."""
    return AA_DICT.get(res, 'X')  # 'X' for unknown residues

'''
def extract_sequences(pdb_list, pdb_dir='./'):
    """
    Extracts sequences from the specified chains in PDB structures.

    :param pdb_list: List of PDB IDs with chain information.
    :param pdb_dir: Directory containing PDB files.
    :return: Dictionary {PDB_ID: {chain: sequence}}
    """
    sequences = {}

    for entry in pdb_list:
        pdb_id = entry[:4]  # Extract PDB ID
        chains = entry[4:]  # Extract chain letters
        
        pdb_file = os.path.join(pdb_dir, f"{pdb_id}.pdb.gz")
        if not os.path.exists(pdb_file):
            LOGGER.info(f"Warning: PDB file {pdb_id} not found! Skipping...")
            continue
        
        structure = parsePDB(pdb_file)
        sequences[pdb_id] = {}

        for chain in chains:
            chain_data = structure.select(f"chain {chain} and protein")
            if chain_data is None:
                LOGGER.info(f"Warning No protein chain {chain} found in {pdb_id}. Skipping...")
                continue

             # Extract residues
            resnums = chain_data.getResnums()
            resnames = chain_data.getResnames()

            # Get unique residue numbers and indices
            unique_resnums, unique_indices = np.unique(resnums, return_index=True)
            unique_resnames = [resnames[i] for i in unique_indices]

            # Convert to 1-letter amino acid sequence
            sequence = "".join([three_to_one(res) for res in unique_resnames])

            # Store sequence and unique residue numbers
            sequences[pdb_id][chain] = (sequence, unique_resnums)

    return sequences
'''

def extract_sequences(pdb_list, pdb_dir='./'):
    """
    Extracts sequences from the specified chains in PDB structures.

    :param pdb_list: List of PDB IDs with chain information.
    :param pdb_dir: Directory containing PDB files.
    :return: Dictionary {PDB_ID: {chain: sequence}}
    """
    sequences = {}

    for entry in pdb_list:
        pdb_id = entry[:4]  # Extract PDB ID
        chains = entry[4:]  # Extract chain letters
        
        pdb_file_1 = os.path.join(pdb_dir, f"{pdb_id}.pdb.gz")
        pdb_file_2 = os.path.join(pdb_dir, f"{entry}")
        
        if os.path.exists(pdb_file_1):
            pdb_file = pdb_file_1
        elif os.path.exists(pdb_file_2):
            pdb_file = pdb_file_2
        else:
            LOGGER.info(f"Warning: PDB entry {entry} not found! Skipping...")
            continue
        
        structure = parsePDB(pdb_file)
        sequences[pdb_id] = {}

        for chain in chains:
            chain_data = structure.select(f"chain {chain} and protein")
            if chain_data is None:
                LOGGER.info(f"Warning No protein chain {chain} found in {pdb_id}. Skipping...")
                continue

             # Extract residues
            resnums = chain_data.getResnums()
            resnames = chain_data.getResnames()

            # Get unique residue numbers and indices
            unique_resnums, unique_indices = np.unique(resnums, return_index=True)
            unique_resnames = [resnames[i] for i in unique_indices]

            # Convert to 1-letter amino acid sequence
            sequence = "".join([three_to_one(res) for res in unique_resnames])

            # Store sequence and unique residue numbers
            sequences[pdb_id][chain] = (sequence, unique_resnums)

    return sequences


# Mapping binding sites to the reference structure
from Bio.PDB import PDBParser, Superimposer, PPBuilder
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
from Bio.Align import substitution_matrices

def decompress_pdb(gz_file_path, output_file_path):
    """ Decompresses a .pdb.gz file and writes it as a .pdb file. """
    with gzip.open(gz_file_path, 'rt', encoding='utf-8') as gz_file, open(output_file_path, 'w') as out_file:
        out_file.write(gz_file.read())

def extract_sequence(structure):
    """ Extracts the full sequence from a PDB structure. """
    ppb = PPBuilder()
    sequence = "".join(str(pp.get_sequence()) for pp in ppb.build_peptides(structure))
    return sequence

def align_sequences(seq1, seq2, matrix='BLOSUM62'):
    """ Aligns two sequences using a substitution matrix (e.g., BLOSUM62). """
    scoring_matrix = substitution_matrices.load(matrix)  # Load BLOSUM62
    alignments = pairwise2.align.globalds(seq1, seq2, scoring_matrix, -10, -0.5)  # Gap penalties

    if not alignments:
        raise ValueError("No valid alignment found!")

    best_alignment = alignments[0]  # Get the best scoring alignment
    aligned_seq1, aligned_seq2, score = best_alignment[:3]  # Extract first three values
    return aligned_seq1, aligned_seq2, score


def find_best_chain_match(reference_pdb, sequences):
    """
    Finds the best matching chain for each PDB by comparing sequences to reference chains.

    Args:
    reference_pdb (str): The PDB ID of the reference structure.
    sequences (dict): Dictionary containing sequence information.

    Returns:
    dict: Mapping of each PDB to the best-matched reference chain.
    """
    reference_chains = sequences[reference_pdb]
    best_matches = {}

    total_pdbs = len(sequences)
    LOGGER.info(f"Starting chain matching for {total_pdbs} PDB structures against reference {reference_pdb}...")

    for pdb_idx, (pdb_id, chains) in enumerate(sequences.items()):
        LOGGER.info(f"Processing PDB {pdb_idx + 1}/{total_pdbs}: {pdb_id}")

        best_chain = None
        best_score = -1
        best_alignment = None

        for chain, (seq, _) in chains.items():
            for ref_chain, (ref_seq, _) in reference_chains.items():
                alignment = align_sequences(ref_seq, seq)

                if alignment and alignment[2] > best_score:
                    best_score = alignment[2]
                    best_chain = (ref_chain, chain)  # (reference_chain, target_chain)
                    best_alignment = alignment

        if best_chain:
            best_matches[pdb_id] = {
                "ref_chain": best_chain[0],
                "target_chain": best_chain[1],
                "alignment": best_alignment
            }
            LOGGER.info(f" Alignment complete: {pdb_id} → {best_chain[1]} (Target) ↔ {best_chain[0]} (Reference) | Score: {best_score}")

    LOGGER.info("Chain matching process completed successfully.")

    return best_matches


def map_binding_sites_to_reference(best_chain_matches, sequences, binding_sites):
    """
    Maps binding sites from other structures to the reference structure based on sequence alignment
    and calculates BLOSUM62 alignment scores.

    Args:
    best_chain_matches (dict): Best matched chain mapping for each PDB.
    sequences (dict): Dictionary containing sequence and residue number information.
    binding_sites (dict): Dictionary containing binding site residues for each PDB.

    Returns:
    dict: Dictionary mapping reference binding sites with total BLOSUM62 scores.
    """
    reference_pdb = list(sequences.keys())[0]  # First PDB as reference
    mapped_sites = {chain: {} for chain in sequences[reference_pdb].keys()}  # Initialize

    # Load BLOSUM62 matrix
    blosum62 = substitution_matrices.load("BLOSUM62")

    LOGGER.info("Reference PDB: " + reference_pdb)
    LOGGER.info("Chains in Reference: " + "".join(list(sequences[reference_pdb].keys())))
    LOGGER.info("Available Binding Sites: " + ", ".join(binding_sites.keys()))

    for pdb_id, match in best_chain_matches.items():
        ref_chain = match["ref_chain"]
        target_chain = match["target_chain"]
        alignment = match["alignment"]

        aligned_ref_seq, aligned_target_seq = alignment[0], alignment[1]
        ref_residues = sequences[reference_pdb][ref_chain]  # (AA sequence, Residue numbers)
        target_residues = sequences[pdb_id][target_chain]

        ref_seq, ref_numbers = ref_residues
        target_seq, target_numbers = target_residues

        # Create mapping between reference and target residues
        ref_to_target = {}
        ref_index = target_index = 0
        for i in range(len(aligned_ref_seq)):
            if aligned_ref_seq[i] != "-":
                if aligned_target_seq[i] != "-":
                    ref_to_target[target_numbers[target_index]] = (ref_seq[ref_index], ref_numbers[ref_index])
                    target_index += 1
                ref_index += 1


        # Process binding sites using **residue numbers only**
        for drug, chains in binding_sites.get(pdb_id, {}).items():
            for target_chain, residues in chains.items():
                if target_chain != match["target_chain"]:
                    continue  # Only consider best-matched chain

                for res in residues:
                    res_num = int(''.join(filter(str.isdigit, res)))  # Extract residue number
                    res_name = ''.join(filter(str.isalpha, res))  # Extract residue name

                    if res_num in ref_to_target:
                        mapped_residue = ref_to_target[res_num]  # (AA name, residue number)
                        ref_residue, ref_res_num = mapped_residue

                        # Compute BLOSUM62 score
                        score = blosum62.get((res_name[0], ref_residue), -4)  # Use first letter of res_name

                        # Store mapped site and accumulate score
                        ref_key = f"{ref_residue}{ref_res_num}"
                        if ref_key not in mapped_sites[ref_chain]:
                            mapped_sites[ref_chain][ref_key] = 0  # Initialize score
                        mapped_sites[ref_chain][ref_key] += score  # Accumulate score


    return mapped_sites
   
def find_similar_chains(reference_pdb, sequences, threshold=0.95):
    """
    Identifies highly similar chains within the reference structure.

    Args:
    reference_pdb (str): Reference PDB ID.
    sequences (dict): Dictionary containing sequence information.
    threshold (float): Sequence similarity threshold (default=0.95).

    Returns:
    list of tuples: Pairs of similar chains (e.g., [('A', 'B'), ('C', 'D')])
    """
    ref_chains = list(sequences[reference_pdb].keys())
    num_chains = len(ref_chains)
    similar_chain_pairs = []

    for i in range(num_chains):
        for j in range(i + 1, num_chains):
            chain1, chain2 = ref_chains[i], ref_chains[j]
            seq1, seq2 = sequences[reference_pdb][chain1][0], sequences[reference_pdb][chain2][0]

            alignment = align_sequences(seq1, seq2)
            identity = alignment[2] / max(len(seq1), len(seq2))  # Compute similarity

            if identity >= threshold:
                similar_chain_pairs.append((chain1, chain2))

    return similar_chain_pairs   


def propagate_binding_sites(mapped_sites, similar_chain_pairs):
    """
    Propagates binding sites and scores between similar chains in the reference structure.

    Args:
    mapped_sites (dict): Dictionary of mapped binding sites and scores for each chain.
    similar_chain_pairs (list of tuples): List of chain pairs with high sequence similarity.

    Returns:
    dict: Updated dictionary with propagated binding sites.
    """
    # Ensure all chains are initialized
    updated_sites = {chain: dict(mapped_sites.get(chain, {})) for chain in mapped_sites}

    for chain1, chain2 in similar_chain_pairs:
        # print(f"Propagating sites between {chain1} and {chain2}")  # Debugging print
        
        # Ensure both chains exist in the dictionary
        if chain1 not in updated_sites:
            updated_sites[chain1] = {}
        if chain2 not in updated_sites:
            updated_sites[chain2] = {}

        # Merge binding site data
        for site, score in mapped_sites.get(chain1, {}).items():
            if site not in updated_sites[chain2]:  # Avoid duplication
                updated_sites[chain2][site] = score

        for site, score in mapped_sites.get(chain2, {}).items():
            if site not in updated_sites[chain1]:  # Avoid duplication
                updated_sites[chain1][site] = score

    converted_binding_sites = convert_binding_sites_to_three_letter(updated_sites)
    return converted_binding_sites


def convert_binding_sites_to_three_letter(mapped_sites):
    """
    Converts one-letter amino acid binding site labels to three-letter labels.

    Args:
    mapped_sites (dict): Dictionary containing mapped binding sites.

    Returns:
    dict: Updated dictionary with three-letter amino acid codes.
    """
    one_to_three = {
        'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
        'E': 'GLU', 'Q': 'GLN', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
        'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
        'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
    }

    updated_sites = {}
    
    for chain, residues in mapped_sites.items():
        updated_sites[chain] = {}
        for res, score in residues.items():
            aa_one = res[0]  # Extract one-letter amino acid
            res_num = res[1:]  # Extract residue number
            
            if aa_one in one_to_three:  # Convert only if it's valid
                aa_three = one_to_three[aa_one]
                updated_residue = f"{aa_three}{res_num}"
                updated_sites[chain][updated_residue] = score
            else:
                LOGGER.info(f"Warning: Unknown residue {aa_one} in {res}")  # Debugging
                updated_sites[chain][res] = score  # Keep original if not found

    return updated_sites


# *************************************************************************************************
# ******************************************* p value *********************************************
# *************************************************************************************************
def parse_hinge_sites(hinge_sites):
    """
    Extracts hinge site residue names, numbers, and chains from hinge site dictionary.

    Args:
    hinge_sites (dict): Dictionary containing primary and minor hinge sites.

    Returns:
    dict: Dictionary {chain: set of (residue, res_num)}
    """
    parsed_hinges = {"primary": {}, "minor": {}}

    for category in ["primary", "minor"]:
        for entry in hinge_sites[category]:
            match = re.match(r"([A-Z]+)(\d+)\((\w)\)", entry)
            if match:
                res_name, res_num, chain = match.groups()
                res_num = int(res_num)
                if chain not in parsed_hinges[category]:
                    parsed_hinges[category][chain] = set()
                parsed_hinges[category][chain].add((res_name, res_num))
    
    return parsed_hinges

def find_overlaps(parsed_hinges, binding_sites):
    """
    Finds overlaps between hinge sites and binding sites.

    Args:
    parsed_hinges (dict): Parsed hinge sites {chain: set of (res_name, res_num)}.
    binding_sites (dict): Dictionary of binding sites {chain: {residue: score}}.

    Returns:
    set: Set of overlapping residues in format (chain, res_num).
    """
    overlaps = set()

    for chain, residues in binding_sites.items():
        for res, _ in residues.items():
            match = re.match(r"([A-Z]+)(\d+)", res)
            if match:
                res_name, res_num = match.groups()
                res_num = int(res_num)

                for category in ["primary", "minor"]:
                    if chain in parsed_hinges[category] and (res_name, res_num) in parsed_hinges[category][chain]:
                        overlaps.add((chain, res_num))

    return overlaps

def filter_binding_sites(binding_sites, overlaps):
    """
    Removes binding sites that do not overlap and have a negative score.

    Args:
    binding_sites (dict): Dictionary {chain: {residue: score}}.
    overlaps (set): Set of overlapping residues (chain, res_num).

    Returns:
    dict: Filtered binding sites.
    """
    filtered_sites = {}

    for chain, residues in binding_sites.items():
        filtered_sites[chain] = {}
        for res, score in residues.items():
            match = re.match(r"([A-Z]+)(\d+)", res)
            if match:
                _, res_num = match.groups()
                res_num = int(res_num)

                if (chain, res_num) in overlaps or score >= 0:
                    filtered_sites[chain][res] = score  # Keep overlapping or positive-score sites

    return filtered_sites

def filter_hinge_sites(parsed_hinges, overlaps):
    """
    Keeps all primary hinge sites and filters minor hinge sites based on overlap.

    Args:
    parsed_hinges (dict): Parsed hinge sites {chain: set of (res_name, res_num)}.
    overlaps (set): Set of overlapping residues (chain, res_num).

    Returns:
    dict: Updated hinge sites with only necessary minor hinges.
    """
    filtered_hinges = {"primary": parsed_hinges["primary"].copy(), "minor": {}}

    for chain, residues in parsed_hinges["minor"].items():
        filtered_hinges["minor"][chain] = {res for res in residues if (chain, res[1]) in overlaps}

    return filtered_hinges

def trim_hinge_sites(filtered_hinges, sequence, overlaps):
    """
    Trims hinge sites from the first and last N/20 residues of each chain unless overlaps exist.

    Args:
    filtered_hinges (dict): Filtered hinge sites.
    sequence (dict): Dictionary of reference sequences.
    overlaps (set): Set of overlapping residues in format like ('B', 68).

    Returns:
    dict: Trimmed hinge site dictionary.
    """
    trimmed_hinges = {"primary": {}, "minor": {}}
    reference_pdb = list(sequence.keys())[0]
    seq_data = sequence[reference_pdb]  # {chain: (res_names, res_ids)}

    for category in ["primary", "minor"]:
        for chain, residues in filtered_hinges[category].items():
            if chain in seq_data:
                res_names, res_ids = seq_data[chain]
                trim_range = len(res_ids) // 20
                lower_bound = res_ids[0] + trim_range
                upper_bound = res_ids[-1] - trim_range

                trimmed_residues = set()
                for res in residues:
                    res_name, res_id = res
                    if (chain, res_id) in overlaps:
                        # Keep overlapping residues
                        trimmed_residues.add(res)
                    elif lower_bound <= res_id <= upper_bound:
                        # Keep residues in the non-terminal range
                        trimmed_residues.add(res)

                trimmed_hinges[category][chain] = trimmed_residues

    return trimmed_hinges

def adjust_hinges_for_binding_sites(trimmed_hinges, binding_sites):
    """
    Adjusts hinge sites if binding sites are in hinge regions.

    Args:
    trimmed_hinges (dict): Trimmed hinge sites.
    binding_sites (dict): Binding sites.

    Returns:
    dict: Updated hinge sites.
    """
    adjusted_hinges = {"primary": {}, "minor": {}}

    for category in ["primary", "minor"]:
        for chain, hinges in trimmed_hinges[category].items():
            hinge_res_nums = {res[1] for res in hinges}
            binding_res_nums = {int(re.search(r"\d+", res).group()) for res in binding_sites.get(chain, {})}

            # Identify continuous hinge regions
            sorted_hinges = sorted(hinge_res_nums)
            hinge_clusters = []
            cluster = []

            for i, res in enumerate(sorted_hinges):
                if i == 0 or res - sorted_hinges[i - 1] == 1:
                    cluster.append(res)
                else:
                    hinge_clusters.append(cluster)
                    cluster = [res]
            if cluster:
                hinge_clusters.append(cluster)

            updated_hinges = set()
            for cluster in hinge_clusters:
                if len(cluster) > 4:  # If more than 4 consecutive hinges
                    mid_cluster = cluster[1:-1]  # Remove first and last hinge
                    lowest_mobility_res = min(mid_cluster)  # Keep lowest mobility residue
                    updated_hinges.update(mid_cluster + [lowest_mobility_res])

                else:
                    updated_hinges.update(cluster)

            # If binding site is in hinge range but not labeled as hinge, include it
            for res in binding_res_nums:
                for cluster in hinge_clusters:
                    if min(cluster) <= res <= max(cluster):
                        updated_hinges.add(res)

            adjusted_hinges[category][chain] = {(res_name, res) for res_name, res in hinges if res in updated_hinges}

    return adjusted_hinges


def process_sites(hinge_sites, binding_sites, sequence, trimmed=False):
    """
    Full pipeline to process hinge and binding sites.

    Args:
    hinge_sites (dict): Hinge site data.
    binding_sites (dict): Binding site data.
    sequence (dict): Sequence mapping.
    trimmed (bool): Whether to trim hinge sites.

    Returns:
    dict: Processed hinge and binding sites.
    """
    parsed_hinges = parse_hinge_sites(hinge_sites)
    overlaps = find_overlaps(parsed_hinges, binding_sites)

    filtered_binding_sites = filter_binding_sites(binding_sites, overlaps)
    filtered_hinges = filter_hinge_sites(parsed_hinges, overlaps)

    if trimmed:
        filtered_hinges = trim_hinge_sites(filtered_hinges, sequence, overlaps)

    adjusted_hinges = adjust_hinges_for_binding_sites(filtered_hinges, filtered_binding_sites)

    return {"hinge_sites": adjusted_hinges, "binding_sites": filtered_binding_sites}

from scipy.stats import hypergeom

# Experiment to evaluate performance of hypergeometric p values
def ORA(M, N, n, k):
    total = 0
    totalLength = N
    binding = M
    hinge = n
    overlap = k
    
    for i in range(overlap):
        total += hypergeom.pmf(i, totalLength, binding, hinge)
    return 1 - total

def write_results_to_file(processed_sites, p_value_result, fileName="site_analysis_results.txt", output_folder="Results"):
    """
    Writes hinge sites, binding sites, overlaps, and p-value to a file in the Result folder.

    Args:
    processed_sites (dict): Dictionary containing hinge sites, binding sites, and overlaps.
    p_value_result (dict): Dictionary containing p-value and statistics.
    output_folder (str): Folder to store the results.
    """

    # Ensure the Result folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    output_file = os.path.join(output_folder, fileName)

    with open(output_file, "w") as f:
        # Write Binding Sites
        f.write("Binding Sites:\n")
        for chain, sites in processed_sites["binding_sites"].items():
            formatted_sites = ", ".join(sorted(sites.keys()))
            f.write(f"Chain {chain}: {formatted_sites}\n")

        f.write("\n")

        # Write Hinge Sites (Merge primary + minor)
        f.write("Hinge Sites:\n")
        merged_hinges = {chain: set() for chain in processed_sites["hinge_sites"]["primary"].keys()}
        
        for category in ["primary", "minor"]:
            for chain, residues in processed_sites["hinge_sites"].get(category, {}).items():
                merged_hinges.setdefault(chain, set()).update(residues)

        for chain, sites in merged_hinges.items():
            formatted_sites = ", ".join(sorted(f"{res[0]}{res[1]}({chain})" for res in sites))
            f.write(f"Chain {chain}: {formatted_sites}\n")

        f.write("\n")

        # Write Overlaps
        f.write("Overlapping Sites:\n")
        overlaps = []
        for chain in merged_hinges.keys():
            chain_overlaps = merged_hinges[chain] & set(
                (res[:3], int(res[3:])) for res in processed_sites["binding_sites"].get(chain, {}).keys()
            )
            overlaps.extend(f"{res[0]}{res[1]}({chain})" for res in chain_overlaps)

        formatted_overlaps = ", ".join(sorted(overlaps))
        f.write(f"{formatted_overlaps}\n\n")

        # Write P-Value Result
        f.write(f"P-value: {p_value_result['p_value']:.6e}\n")
        f.write(f"Total Residues (N): {p_value_result['N']}\n")
        f.write(f"Binding Sites (M): {p_value_result['M']}\n")
        f.write(f"Hinge Sites (n): {p_value_result['n']}\n")
        f.write(f"Overlapping Sites (k): {p_value_result['k']}\n")

    LOGGER.info(f"Results written to: {output_file}")


def compute_p_value(processed_sites, sequence_data, fileName="Drug_vs_Hinges.txt", ref=None):
    """
    Computes the over-representation analysis (ORA) p-value using the hypergeometric test.

    Args:
    processed_sites (dict): Dictionary containing hinge sites, binding sites, and overlaps.
    sequence_data (dict): Dictionary containing sequence information for the reference PDB.

    Returns:
    dict: Dictionary containing p-value and key statistics.
    """
    # Extract reference PDB ID (first key in sequence data)
    reference_pdb = list(sequence_data.keys())[0]

    # Get the total number of residues in the reference structure (N)
    if ref == None:
        total_residues = sum(len(seq_data[1]) for seq_data in sequence_data[reference_pdb].values())
    else: 
        total_residues = len(ref.getResnames())
    # Merge primary and minor hinge sites
    hinge_sites = {}
    for category in ["primary", "minor"]:
        for chain, residues in processed_sites["hinge_sites"].get(category, {}).items():
            hinge_sites.setdefault(chain, set()).update(residues)

    # Convert hinge site format (e.g., ('ALA', 431) -> 'ALA431(A)')
    hinge_set = {f"{res}{num}({chain})" for chain, sites in hinge_sites.items() for res, num in sites}

    # Convert binding site format (e.g., {'ALA527': 4.0} -> 'ALA527(A)')
    binding_set = {f"{res[:3]}{num}({chain})" for chain, sites in processed_sites["binding_sites"].items()
                   for res, num in (zip(sites.keys(), [s[3:] for s in sites.keys()]))}

    # Get the number of binding sites (M)
    binding_count = len(binding_set)

    # Get the number of hinge sites (n)
    hinge_count = len(hinge_set)

    # Get the number of overlaps (k)
    overlaps = len(hinge_set & binding_set)

    # Compute hypergeometric p-value
    p_value = ORA(binding_count, total_residues, hinge_count, overlaps)

    # Ensure we pass a dictionary with the correct values
    p_value_result = {
        "p_value": p_value,
        "N": total_residues,
        "M": binding_count,
        "n": hinge_count,
        "k": overlaps
    }

    # Write results to file
    write_results_to_file(processed_sites, p_value_result, list(sequence_data.keys())[0] + '_' + fileName)

    return p_value_result       


# *************************************************************************************************
# ******************************************* Plotting ********************************************
# *************************************************************************************************
def map_residues_to_indices(binding_sites, sequence):
    """
    Maps residue labels to sequential indices for plotting.

    Args:
    - binding_sites (set of str): Set like {'GLY193(A)', ...}
    - hinge_sites (dict): Dict with 'primary' and 'minor' lists of residue strings
    - sequence (dict): Sequence dict with format {'4m11': {'A': (res_names, res_ids), 'B': ...}}
    - trim_ends (bool): If True, trims hinge residues at ends unless overlapping
    - trim_fraction (float): Fraction of residues at each end to consider as trim zones

    Returns:
    - dict: {"binding_indices": set of ints, "hinge_indices": {"primary": set, "minor": set}}
    - dict: reverse_index_map from residue label (e.g., 'GLY193(A)') → index
    """

    reference_pdb = list(sequence.keys())[0]
    chain_seq = sequence[reference_pdb]  # e.g., {'A': (res_names, res_ids), ...}

    index_map = {}
    reverse_index_map = {}
    running_index = 0

    # Step 1: Build residue-to-index mapping
    for chain_id in sorted(chain_seq.keys()):
        res_names, res_ids = chain_seq[chain_id]
        for i, res_num in enumerate(res_ids):
            res_name = res_names[i]
            res_str = f"{res_name}{res_num}({chain_id})"
            index_map[(chain_id, res_num)] = running_index
            reverse_index_map[res_str] = running_index
            running_index += 1

    # Step 2: Map binding site strings to indices
    binding_indices = set()
    overlap_coords = set()
    for res_str in binding_sites:
        match = re.match(r"([A-Z]{3})(\d+)\(([A-Z])\)", res_str)
        if match:
            resname, resid, chain = match.groups()
            resid = int(resid)
            key = (chain, resid)
            if key in index_map:
                idx = index_map[key]
                binding_indices.add(idx)
                overlap_coords.add(key)

    return binding_indices, reverse_index_map
    
def get_final_hinge_list(binding_index, primary_regions, minor_regions):
    """
    Combines primary hinge regions with minor hinge residues that overlap with binding sites.

    Args:
    - binding_index (set of int): Binding site indices
    - primary_regions (list of lists of int): Primary hinge indices
    - minor_regions (list of lists of int): Minor hinge indices

    Returns:
    - list of int: Combined final hinge list
    """
    # Flatten nested lists
    flat_primary = {res for group in primary_regions for res in group}
    flat_minor = {res for group in minor_regions for res in group}

    # Include minor residues that overlap with binding index
    final_hinges = flat_primary.union(flat_minor.intersection(binding_index))

    return sorted(set(final_hinges))

def trim_hinge_indices(hinge_indices, binding_indices, sequence, ref_structure):
    """
    Trim hinge indices at the chain ends unless they overlap with binding sites.

    Args:
        hinge_indices (list of int): List of all hinge indices (global indices across chains).
        binding_indices (set of int): Set of indices that are binding sites.
        sequence (dict): Sequence dictionary {ref_structure: {chain: (res_names, res_ids)}}
        ref_structure (str): Key of the reference structure (e.g., "4m11")

    Returns:
        list of int: Trimmed hinge indices
    """
    # Build global index lookup for each chain
    chain_index_ranges = {}
    global_index = 0

    for chain_id, (res_names, res_nums) in sequence[ref_structure].items():
        length = len(res_nums)
        indices = list(range(global_index, global_index + length))
        chain_index_ranges[chain_id] = indices
        global_index += length

    # Trim logic per chain
    hinge_indices_set = set(hinge_indices)
    final_hinges = set()

    for chain_id, indices in chain_index_ranges.items():
        trim_len = max(1, len(indices) // 20)
        trim_start = set(indices[:trim_len])
        trim_end = set(indices[-trim_len:])

        for idx in indices:
            if idx not in hinge_indices_set:
                continue
            # Trim if in start/end AND not in binding
            if (idx in trim_start or idx in trim_end) and idx not in binding_indices:
                continue
            final_hinges.add(idx)

    return sorted(final_hinges)

'''
def plot_hinge_binding_mode_custom(
    mode,
    gnms,
    sequences,
    hinge_residues,
    converted_binding_sites,
    threshold=15,
    trim=False,
    hinge_marker='o',
    hinge_color=(0.5, 1, 0),
    hinge_size=15,
    binding_marker='*',
    binding_color='r',
    binding_size=18,
    overlap_marker='D',
    overlap_color='m',
    overlap_size=18,
    font_size=50,
    tick_size=40,
    figsize=(18, 8),
    showing=True,
    save=False,
    save_dir='Results'
):
    """
    Flexible GNM mode visualization with hinge and binding site overlays.

    Args:
        mode (int): GNM mode index.
        gnms: GNM eigenvector object (N x m).
        avg_eigvecs (np.ndarray): Mean eigenvectors (N x m).
        sequences (dict): Sequence mapping from process.
        converted_binding_sites (dict): Binding site data.
        threshold (float): Threshold for hinge detection.
        trimmed (bool): Whether to trim hinge ends.
        *marker, *color, *size: Custom plot styles for each residue type.
        font_size, tick_size: Font/tick control.
        figsize (tuple): Plot size.
        show (bool): Whether to call show() at the end.
        save (bool): Whether to save the figure to disk.
        save_dir (str): Output directory for saved figures.
    """
    # Step 1: Data preparing
    # prepare data
    eigVects = gnms.getEigvecs()
    avg_eigvecs = mean(eigVects, axis=0)
    
    primary_hinges = set()
    minor_hinges = set()
    refStructure = list(sequences.keys())[0]
    refChain = "".join(list(sequences[refStructure].keys()))
    
    primary_regions, minor_regions = detectSingleModeHinges(avg_eigvecs[:, mode], threshold=threshold)
    primary_hinges.update(flattenArray(primary_regions))
    minor_hinges.update(flattenArray(minor_regions))
    processed = process_sites(hinge_residues, converted_binding_sites, sequences)
    
    # binding sites
    binding_set = {f"{res[:3]}{num}({chain})" for chain, sites in processed["binding_sites"].items()
                   for res, num in (zip(sites.keys(), [s[3:] for s in sites.keys()]))}
    binding_index, residue_to_index = map_residues_to_indices(binding_sites=binding_set, sequence=sequences)
    
    # hinges
    hinge_list = get_final_hinge_list(binding_index, primary_regions, minor_regions)
    # Trim hinge list based on chain ends
    if trim:
        hinge_list = trim_hinge_indices(
            hinge_indices=hinge_list,
            binding_indices=binding_index,
            sequence=sequences,
            ref_structure=refStructure
        )

    # trim and find overlaps
    currHinge = list(hinge_list)
    binding = list(binding_index)
    overlaps = [i for i in binding if i in currHinge]
    
    # Step 2: Plot
    fig, ax = subplots(figsize=figsize)
    sca(ax)
    ax.set_yticks([])
    
    # Set tick label font sizes
    rc('xtick', labelsize=tick_size)
    rc('ytick', labelsize=tick_size)
    
    # Set tick label font sizes
    # ax.tick_params(axis='x', labelsize=tick_size)
    # ax.tick_params(axis='y', labelsize=tick_size)
    
    rcParams.update({'font.size': font_size})

    title(f'Mode {mode + 1}', fontweight="bold")

    showSignatureMode(gnms[:, mode], linewidth=3)

    # Plot hinges
    HingeY = [avg_eigvecs[i, mode] for i in currHinge]
    plot(currHinge, HingeY, color=hinge_color, marker=hinge_marker, linestyle='', markersize=hinge_size)

    # Plot bindings
    bindingY = [avg_eigvecs[i, mode] for i in binding]
    plot(binding, bindingY, marker=binding_marker, color=binding_color, linestyle='', markersize=binding_size)

    # Plot overlaps
    overlapsY = [avg_eigvecs[i, mode] for i in overlaps]
    plot(overlaps, overlapsY, marker=overlap_marker, color=overlap_color, linestyle='', markersize=overlap_size)

    title(f'Mode {mode + 1}', fontweight="bold")

    # prin the results
    # Reverse the residue_to_index dictionary
    index_to_residue = {v: k for k, v in residue_to_index.items()}
    
    # Helper to convert list of indices to comma-separated residue strings
    def format_residues(index_list):
        return ", ".join(index_to_residue.get(idx, f"Index {idx} not found") for idx in sorted(index_list))

    binding_str = format_residues(binding)
    hinge_str = format_residues(currHinge)
    overlap_str = format_residues(overlaps)
    
    # Print outputs
    LOGGER.info("Binding Residues:")
    LOGGER.info(binding_str)
    
    LOGGER.info("\nHinge Residues:")
    LOGGER.info(hinge_str)
    
    LOGGER.info("\nOverlapping Residues:")
    LOGGER.info(overlap_str)

    
    # Save figure if requested
    if save:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"{refStructure}_mode{mode + 1}.png")
        tight_layout()
        savefig(out_path, bbox_inches='tight')

        # Save text file
        txt_path = os.path.join(save_dir, f"{refStructure}_mode{mode + 1}.txt")
        with open(txt_path, 'w') as f:
            f.write("Binding Residues:\n")
            f.write(binding_str + "\n\n")
            f.write("Hinge Residues:\n")
            f.write(hinge_str + "\n\n")
            f.write("Overlapping Residues:\n")
            f.write(overlap_str + "\n")

        
        LOGGER.info(f"Figure saved to: {out_path}")

    if showing:
        show()
'''

def plot_hinge_binding_mode_custom(
    mode,
    gnms,
    sequences,
    hinge_residues,
    converted_binding_sites,
    threshold=15,
    trim=False,
    hinge_marker='o',
    hinge_color=(0.5, 1, 0),
    hinge_size=15,
    binding_marker='*',
    binding_color='r',
    binding_size=18,
    overlap_marker='D',
    overlap_color='m',
    overlap_size=18,
    font_size=50,
    tick_size=40,
    figsize=(18, 8),
    showing=True,
    save=False,
    save_dir='Results'
):
    """
    Flexible GNM mode visualization with hinge and binding site overlays.

    Args:
        mode (int): GNM mode index.
        gnms: GNM eigenvector object (N x m).
        avg_eigvecs (np.ndarray): Mean eigenvectors (N x m).
        sequences (dict): Sequence mapping from process.
        converted_binding_sites (dict): Binding site data.
        threshold (float): Threshold for hinge detection.
        trimmed (bool): Whether to trim hinge ends.
        *marker, *color, *size: Custom plot styles for each residue type.
        font_size, tick_size: Font/tick control.
        figsize (tuple): Plot size.
        show (bool): Whether to call show() at the end.
        save (bool): Whether to save the figure to disk.
        save_dir (str): Output directory for saved figures.
    """
    # Step 1: Data preparing
    # prepare data
    eigVects = gnms.getEigvecs()
    avg_eigvecs = mean(eigVects, axis=0)
    
    primary_hinges = set()
    minor_hinges = set()
    refStructure = list(sequences.keys())[0]
    refChain = "".join(list(sequences[refStructure].keys()))
    
    primary_regions, minor_regions = detectSingleModeHinges(avg_eigvecs[:, mode], threshold=threshold)
    primary_hinges.update(flattenArray(primary_regions))
    minor_hinges.update(flattenArray(minor_regions))
    processed = process_sites(hinge_residues, converted_binding_sites, sequences)
    
    # binding sites
    binding_set = {f"{res[:3]}{num}({chain})" for chain, sites in processed["binding_sites"].items()
                   for res, num in (zip(sites.keys(), [s[3:] for s in sites.keys()]))}
    binding_index, residue_to_index = map_residues_to_indices(binding_sites=binding_set, sequence=sequences)
    
    # hinges
    hinge_list = get_final_hinge_list(binding_index, primary_regions, minor_regions)
    # Trim hinge list based on chain ends
    if trim:
        hinge_list = trim_hinge_indices(
            hinge_indices=hinge_list,
            binding_indices=binding_index,
            sequence=sequences,
            ref_structure=refStructure
        )

    # trim and find overlaps
    currHinge = list(hinge_list)
    binding = list(binding_index)
    overlaps = [i for i in binding if i in currHinge]
    
    # Step 2: Plot
    fig, ax = subplots(figsize=figsize)
    sca(ax)
    ax.set_yticks([])
    
    # Set tick label font sizes
    rc('xtick', labelsize=tick_size)
    rc('ytick', labelsize=tick_size)
    
    # Set tick label font sizes
    ax.tick_params(axis='x', labelsize=tick_size)
    ax.tick_params(axis='y', labelsize=tick_size)
    
    rcParams.update({'font.size': font_size})

    title(f'Mode {mode + 1}', fontweight="bold")
    ax.set_title(f'Mode {mode + 1}', fontsize=font_size, fontweight='bold')
    
    ax.set_xlabel("Residue Index", fontsize=font_size)
    
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(tick_size)
    
    showSignatureMode(gnms[:, mode], linewidth=3)

    # Plot hinges
    HingeY = [avg_eigvecs[i, mode] for i in currHinge]
    plot(currHinge, HingeY, color=hinge_color, marker=hinge_marker, linestyle='', markersize=hinge_size)

    # Plot bindings
    bindingY = [avg_eigvecs[i, mode] for i in binding]
    plot(binding, bindingY, marker=binding_marker, color=binding_color, linestyle='', markersize=binding_size)

    # Plot overlaps
    overlapsY = [avg_eigvecs[i, mode] for i in overlaps]
    plot(overlaps, overlapsY, marker=overlap_marker, color=overlap_color, linestyle='', markersize=overlap_size)

    title(f'Mode {mode + 1}', fontweight="bold")

    # prin the results
    # Reverse the residue_to_index dictionary
    index_to_residue = {v: k for k, v in residue_to_index.items()}
    
    # Helper to convert list of indices to comma-separated residue strings
    def format_residues(index_list):
        return ", ".join(index_to_residue.get(idx, f"Index {idx} not found") for idx in sorted(index_list))

    binding_str = format_residues(binding)
    hinge_str = format_residues(currHinge)
    overlap_str = format_residues(overlaps)
    
    # Print outputs
    LOGGER.info("Binding Residues:")
    LOGGER.info(binding_str)
    
    LOGGER.info("\nHinge Residues:")
    LOGGER.info(hinge_str)
    
    LOGGER.info("\nOverlapping Residues:")
    LOGGER.info(overlap_str)

    
    # Save figure if requested
    if save:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"{refStructure}_mode{mode + 1}.png")
        tight_layout()
        savefig(out_path, bbox_inches='tight')

        # Save text file
        txt_path = os.path.join(save_dir, f"{refStructure}_mode{mode + 1}.txt")
        with open(txt_path, 'w') as f:
            f.write("Binding Residues:\n")
            f.write(binding_str + "\n\n")
            f.write("Hinge Residues:\n")
            f.write(hinge_str + "\n\n")
            f.write("Overlapping Residues:\n")
            f.write(overlap_str + "\n")

        
        LOGGER.info(f"Figure saved to: {out_path}")

    if showing:
        show()
        
        

# Alter B factors using GNM model
# replace b factors using eigenvectors
def write_motion_from_sequences(sequences, avg_eigvecs, mode, ref=None, output_dir='structure_demo'):
    os.makedirs(output_dir, exist_ok=True)

    ref_name = list(sequences.keys())[0]
    ref_struct = sequences[ref_name]

    # If ref is given (e.g., calphas), use its chain and resnums directly
    if ref is not None:
        resiIndex = ref.getResnums().tolist()
        chain = ref.getChids().tolist()
    else:
        # Use full sequence from both chains
        resiIndex = []
        chain = []
        for chain_id, (seq, res_array) in ref_struct.items():
            resiIndex.extend(res_array.tolist())
            chain.extend([chain_id] * len(res_array))

    # Sanity check
    if len(resiIndex) != avg_eigvecs.shape[0]:
        raise ValueError(f"Mismatch: {len(resiIndex)} residues vs {avg_eigvecs.shape[0]} eigenvector entries")

    # Extract values for the selected mode
    new_b_factor = avg_eigvecs[:, mode].tolist()

    # Output file
    out_file = os.path.join(output_dir, f"{ref_name}_b_mode{mode + 1}.txt")
    with open(out_file, 'w') as wf:
        for ch, resi, motion in zip(chain, resiIndex, new_b_factor):
            wf.write(f"{ch}, {resi}, {motion}\n")

    LOGGER.info(f"Mode {mode + 1} written to: {out_file}")
    return out_file

def generate_pymol_bfactor_script(structure_pdb: str, bfactor_txt: str, output_pdb: str, output_dir: str = "structure_demo"):
    """
    Writes a .pml script inside `output_dir` to:
    - Load a PDB file (located in `output_dir`)
    - Apply B-factors from `bfactor_txt` (also in `output_dir`)
    - Save the updated structure to `output_pdb` in `output_dir`

    All paths inside the script are relative (no `output_dir/`), since PyMOL is run from that folder.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Base file names only (since we run from inside the output_dir)
    structure_name = os.path.basename(structure_pdb)
    bfactor_file = os.path.basename(bfactor_txt)
    output_pdb_name = os.path.basename(output_pdb)

    # Path to save .pml file (in output_dir)
    pml_path = os.path.join(output_dir, f"{structure_name.replace('.pdb', '')}_set_bfactor.pml")

    with open(pml_path, 'w') as f:
        f.write(f"# Load structure\n")
        f.write(f"load {structure_name}\n\n")

        f.write(f"# Apply B-factors from file\n")
        f.write(f"@{bfactor_file}\n\n")

        f.write("python\n")
        f.write(f"with open('{bfactor_file}', 'r') as file:\n")
        f.write("    for line in file:\n")
        f.write("        if line.startswith('#') or not line.strip():\n")
        f.write("            continue\n")
        f.write("        chain, residue_id, new_b_factor = line.split(',')\n")
        f.write("        chain = chain.strip()\n")
        f.write("        residue_id = residue_id.strip()\n")
        f.write("        new_b_factor = float(new_b_factor.strip())\n")
        f.write(f"        selection = f'/{structure_name.replace('.pdb','')}//{{chain}}/{{residue_id}}/'\n")
        f.write("        cmd.alter(selection, f'b={new_b_factor}')\n")
        f.write("cmd.rebuild()\n")
        f.write("python end\n\n")

        f.write(f"# Save modified structure\n")
        f.write(f"save {output_pdb_name}\n")

    LOGGER.info(f"PyMOL script written to {pml_path} (run from inside {output_dir})")
    
