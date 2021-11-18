
The full python workflow for semasiological token-level clouds
==============================================================

This notebook shows the full workflow followed to create the clouds in
Chapter 5 of the monograph (and my dissertation).

(Eventually, add Kris's chapter too)

For this purpose I'm creating a Python package with more order than my
previous attempts. It is called ``semasioFlow`` and has the most verbose
code; its documentation can be explored
`here <docs/build/html/index.html>`__.

0. Initial setup
----------------

So first we have to add this directory to the path in order to import
``semasioFlow``. When it initializes, it loads other libraries we need
as well as the ``qlvl`` library from the ``depmodel`` repository.

.. code:: ipython3

    import os
    import sys
    import logging
    sys.path.append("/home/aardvark/code/nephosem")
    mypackage = "/home/projects/semmetrix/mariana_wolken/cleanWorkflow/scripts/"
    sys.path.append(mypackage)

.. code:: ipython3

    from semasioFlow import ConfigLoader
    from semasioFlow.load import loadVocab, loadMacro, loadColloc, loadFocRegisters
    from semasioFlow.sample import sampleTypes
    from semasioFlow.focmodels import createBow, createRel, createPath
    from semasioFlow.socmodels import targetPPMI, weightTokens, createSoc
    from semasioFlow.utils import plotPatterns

1. Configuration
----------------

Depending on what you need, you will have to set up some useful paths
settings. I like to have at least the path to my project (``mydir``), an
output path within (``mydir + "output"``) and a GitHub path for the
datasets that I will use in the visualization. There is no real reason
not to have everything together, except that I did not think of it at
the moment. (Actually, there is: the GitHub stuff will be public and
huge data would not be included. How much do we want to have public?)

.. code:: ipython3

    mydir = "/home/projects/semmetrix/mariana_wolken/cleanWorkflow/"
    output_path = f"{mydir}/output/"
    github_dir = f"{mydir}/github/"
    logging.basicConfig(filename = f'{mydir}/testlog.log', level = logging.DEBUG)

The variables with paths is just meant to make it easier to manipulate
filenames. The most important concrete step is to adapt the
configuration file.

.. code:: ipython3

    conf = ConfigLoader()
    default_settings = conf.settings
    settings = default_settings
    # Regular expression to capture data from the QLVLNewsCorpus
    settings['line-machine'] = '([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t_\t_'
    settings['separator-line-machine'] = "^</sentence>$"
    settings['global-columns'] = "id,word,lemma,pos,head,deprel"
    settings['note-attr'] = 'word,lemma,pos'
    settings['edge-attr'] = 'deprel'
    settings['currID'] = 'id'
    settings['headID'] = 'head'
    settings['type'] = 'lemma/pos'
    settings['colloc'] = 'lemma/pos'
    settings['token'] = 'lemma/pos/fid/lid'
    
    settings['file-encoding'] = 'latin1'
    settings['outfile-encoding'] = 'utf-8'
    settings['output-path'] = output_path
    settings['corpus-path'] = '/home/aardvark/corp/nl/'
    
    print(settings['line-machine'])
    print(settings['global-columns'])
    print(settings['type'], settings['colloc'], settings['token'])


.. parsed-literal::

    ([^	]+)	([^	]+)	([^	]+)
    word,pos,lemma
    lemma/pos lemma/pos lemma/pos/fid/lid


2. Frequency lists
------------------

The frequency lists are the first thing to create, but once you have
them, you just load them. So what we are going to do here is define the
filename where we *would* store the frequency list (in this case, where
it is actually stored), and if it exists it loads it; if it doesn't, it
creates and store it.

.. code:: ipython3

    full_name = f"{output_path}/vocab/QLVLNews.nodefreq"
    print(full_name)
    full = loadVocab(full_name, settings)
    full


.. parsed-literal::

    /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//vocab/QLVLNews.nodefreq




.. parsed-literal::

    [('./punct', 28225482),('de/det', 28074105),(',/punct', 18092245) ... ('België-VTS/name', 1),('fertility/name', 1),('appelerend/adj', 1)]



.. code:: ipython3

    foc_name = f"{output_path}/vocab/QLVLNews.FOC.nodefreq"
    print(foc_name)
    foc = loadVocab(foc_name, settings)
    foc


.. parsed-literal::

    /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//vocab/QLVLNews.FOC.nodefreq




.. parsed-literal::

    [('de/det', 28074105),('van/prep', 12638933),('het/det', 11650779) ... ('Uitbergen/name', 227),('provincie_weg/noun', 227),('Jommeke/name', 227)]



.. code:: ipython3

    soc_name = f"{output_path}/vocab/QLVLNews.contextwords_final.nodefreq"
    print(soc_name)
    soc = loadVocab(soc_name, settings)
    soc


.. parsed-literal::

    /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//vocab/QLVLNews.contextwords_final.nodefreq




.. parsed-literal::

    [('ben/verb', 9938754),('heb/verb', 4034811),('word/verb', 3486710) ... ('intergouvernementeel/adj', 337),('verdrinking/noun', 337),('popcorn/noun', 336)]



3. Boolean token-level matrices
-------------------------------

Even though we first think of the type leven and only afterwards of the
token level, with this workflow we don't really need to touch type level
until after we obtain the boolean token-level matrices, that is, until
we need to use PPMI values to select or weight the context words.

As a first step, we need the type or list of types we want to run; for
example ``"heet/adj"`` or ``["vernietig/verb", "verniel/verb"]``, and we
subset the vocabulary for that query.

.. code:: ipython3

    fnames = f"{mydir}/sources/LeNC260TwNC260.fnames"

.. code:: ipython3

    with open(fnames, 'r') as f:
        fname_list = [x.strip() for x in f.readlines()]
    fname_list
    NL = [x for x in fname_list if x.startswith('/home/aardvark/corp/nl/TwNC')]
    BE = [x for x in fname_list if x.startswith('/home/aardvark/corp/nl/LeNC')]
    tokenlist_NL, fnameSample_NL = sampleTypes(..., fnames = NL)
    tokenlist_BE, fnameSample_BE = sampleTypes(..., fnames = BE)
    tokenlist = tokenlistNL + tokenlist_BE
    fnameSample = fnameSample_NL + fnameSample_BE




.. parsed-literal::

    set()



.. code:: ipython3

    query = full.subvocab(["verniel/verb", "vernietig/verb"])
    type_name = "destroy"
    query




.. parsed-literal::

    [('verniel/verb', 7507),('vernietig/verb', 12128)]



We could generate the tokens for all 10k tokens, or create a random
selection with a certain number and then only use those files. The
output of sampleTypes includes a list of token IDs as well as the list
of filenames that suffices to extract those tokens. We can then use the
new list of filenames when we collect tokens, and the list of tokens to
subset the resulting matrices.

Of course, to keep the sample fixed it would be more useful to generate
the list, store it and then retrieve it in future runs.

.. code:: ipython3

    import json
    import os.path
    
    #tokenlist_fname = f"{mydir}/sources/destroy.json" # original file
    tokenlist_fname = f"{mydir}/sources/filelist2.json" # subset of 8
    if os.path.exists(tokenlist_fname):
        with open(tokenlist_fname, "r") as f:
            tokenlist, fnameSample = json.load(f).values()
    else:
        tokenlist, fnameSample = sampleTypes({"verniel/verb" : 30, "vernietig/verb" : 30}, fnames)
        with open(tokenlist_fname, "w") as f:
            json.dump({"tokenlist" : tokenlist, "fnames" : fnameSample}, f)

.. code:: ipython3

    len(tokenlist)




.. parsed-literal::

    60



3.1 Bag-of-words
~~~~~~~~~~~~~~~~

The code to generate one matrix is very straightforward, but what if we
want to use different combinations of parameter settings to create
multiple matrices?

The code below assumes that the boolean BOW matrices may vary across
three parameters:

-  **foc\_win**: window size, which is set with numbers for let and
   right window. *This has the settings above for default*
-  **foc\_pos**: part-of-speech filter, which will actually be set as a
   previously filtered list of context words. *By default, all context
   words are included.*
-  **bound**: the match for sentence boundaries and whether the models
   respond to them or not. *By default, sentence boundaries are
   ignored.*

.. code:: ipython3

    lex_pos = [x for x in foc.get_item_list() if x.split("/")[1] in ["noun", "adj", "verb", "adv"]]

.. code:: ipython3

    foc_win = [(3, 3), (5, 5), (10, 10)] #three options of symmetric windows with 3, 5 or 10 words to each side
    foc_pos = {
        "all" : foc.get_item_list(), # the filter has already been applied in the FOC list
        "lex" : lex_pos # further filter by part-of-speech
    }
    bound = { "match" : "^</sentence>$", "values" : [True, False]}

The function below combines a number of necessary functions:

-  it creates a loop over the different combinations of parameter
   settings specified
-  it collects the tokens and computes and filters the corresponding
   matrices
-  it transforms the matrices in "boolean" integer matrices, with only
   0's and 1's
-  it stores the matrices in their respective files
-  it records the combinations of parameter settings and which values
   are taken by each model
-  it records the context words captured by each model for each token
-  it returns both records to be stored wherever you want

.. code:: ipython3

    bowdata = createBow(query, settings, type_name = type_name, fnames = fnameSample, tokenlist = tokenlist,
            foc_win = foc_win, foc_pos = foc_pos, bound = bound)
    bowdata.to_csv(f"{output_path}/registers/{type_name}.bow-models.tsv", sep = "\t", index_label = "_model")


.. parsed-literal::

      0%|          | 0/6 [00:00<?, ?it/s]

.. parsed-literal::

    WARNING: Not provide the temporary path!
    WARNING: Use the default tmp directory: '~/tmp'!
    Scanning tokens of queries in corpus...
    WARNING: 60381 columns have not been found.
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy//destroy.bound3-3all.tcmx.bool.pac
    WARNING: 35742 columns have not been found.
    
    Saving matrix...


.. parsed-literal::

     17%|█▋        | 1/6 [00:00<00:04,  1.11it/s]

.. parsed-literal::

    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy//destroy.bound3-3lex.tcmx.bool.pac
    WARNING: Not provide the temporary path!
    WARNING: Use the default tmp directory: '~/tmp'!
    Scanning tokens of queries in corpus...
    WARNING: 60348 columns have not been found.
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy//destroy.nobound3-3all.tcmx.bool.pac
    WARNING: 35726 columns have not been found.
    
    Saving matrix...


.. parsed-literal::

     33%|███▎      | 2/6 [00:01<00:03,  1.14it/s]

.. parsed-literal::

    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy//destroy.nobound3-3lex.tcmx.bool.pac
    WARNING: Not provide the temporary path!
    WARNING: Use the default tmp directory: '~/tmp'!
    Scanning tokens of queries in corpus...
    WARNING: 60300 columns have not been found.
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy//destroy.bound5-5all.tcmx.bool.pac
    WARNING: 35686 columns have not been found.
    
    Saving matrix...


.. parsed-literal::

     50%|█████     | 3/6 [00:02<00:02,  1.11it/s]

.. parsed-literal::

    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy//destroy.bound5-5lex.tcmx.bool.pac
    WARNING: Not provide the temporary path!
    WARNING: Use the default tmp directory: '~/tmp'!
    Scanning tokens of queries in corpus...
    WARNING: 60238 columns have not been found.
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy//destroy.nobound5-5all.tcmx.bool.pac
    WARNING: 35645 columns have not been found.
    
    Saving matrix...


.. parsed-literal::

     67%|██████▋   | 4/6 [00:03<00:01,  1.13it/s]

.. parsed-literal::

    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy//destroy.nobound5-5lex.tcmx.bool.pac
    WARNING: Not provide the temporary path!
    WARNING: Use the default tmp directory: '~/tmp'!
    Scanning tokens of queries in corpus...
    WARNING: 60166 columns have not been found.
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy//destroy.bound10-10all.tcmx.bool.pac
    WARNING: 35591 columns have not been found.
    
    Saving matrix...


.. parsed-literal::

     83%|████████▎ | 5/6 [00:04<00:00,  1.11it/s]

.. parsed-literal::

    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy//destroy.bound10-10lex.tcmx.bool.pac
    WARNING: Not provide the temporary path!
    WARNING: Use the default tmp directory: '~/tmp'!
    Scanning tokens of queries in corpus...
    WARNING: 60013 columns have not been found.
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy//destroy.nobound10-10all.tcmx.bool.pac
    WARNING: 35487 columns have not been found.
    
    Saving matrix...


.. parsed-literal::

    100%|██████████| 6/6 [00:05<00:00,  1.10it/s]

.. parsed-literal::

    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy//destroy.nobound10-10lex.tcmx.bool.pac


.. parsed-literal::

    


.. code:: ipython3

    bowdata




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>bound</th>
          <th>foc_base</th>
          <th>foc_pos</th>
          <th>foc_win</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>destroy.bound10-10all</th>
          <td>True</td>
          <td>BOW</td>
          <td>all</td>
          <td>10-10</td>
        </tr>
        <tr>
          <th>destroy.bound10-10lex</th>
          <td>True</td>
          <td>BOW</td>
          <td>lex</td>
          <td>10-10</td>
        </tr>
        <tr>
          <th>destroy.bound3-3all</th>
          <td>True</td>
          <td>BOW</td>
          <td>all</td>
          <td>3-3</td>
        </tr>
        <tr>
          <th>destroy.bound3-3lex</th>
          <td>True</td>
          <td>BOW</td>
          <td>lex</td>
          <td>3-3</td>
        </tr>
        <tr>
          <th>destroy.bound5-5all</th>
          <td>True</td>
          <td>BOW</td>
          <td>all</td>
          <td>5-5</td>
        </tr>
        <tr>
          <th>destroy.bound5-5lex</th>
          <td>True</td>
          <td>BOW</td>
          <td>lex</td>
          <td>5-5</td>
        </tr>
        <tr>
          <th>destroy.nobound10-10all</th>
          <td>False</td>
          <td>BOW</td>
          <td>all</td>
          <td>10-10</td>
        </tr>
        <tr>
          <th>destroy.nobound10-10lex</th>
          <td>False</td>
          <td>BOW</td>
          <td>lex</td>
          <td>10-10</td>
        </tr>
        <tr>
          <th>destroy.nobound3-3all</th>
          <td>False</td>
          <td>BOW</td>
          <td>all</td>
          <td>3-3</td>
        </tr>
        <tr>
          <th>destroy.nobound3-3lex</th>
          <td>False</td>
          <td>BOW</td>
          <td>lex</td>
          <td>3-3</td>
        </tr>
        <tr>
          <th>destroy.nobound5-5all</th>
          <td>False</td>
          <td>BOW</td>
          <td>all</td>
          <td>5-5</td>
        </tr>
        <tr>
          <th>destroy.nobound5-5lex</th>
          <td>False</td>
          <td>BOW</td>
          <td>lex</td>
          <td>5-5</td>
        </tr>
      </tbody>
    </table>
    </div>



3.2 Lemmarel
~~~~~~~~~~~~

For dependency models we need specific templates and and patterns ---
especially for LEMMAREL, they need to be tailored to the part-of-speech
that you are looking into. Since I'm exemplifying with a verb, I will
use those templates.

**IMPORTANT**: In order to work, dependency models require the
'separator-line-machine' value.

**Note**: The old code used a lot of upper case; these copies of
templates use only lower case. I will soon fix that in the other
templates and move them to this directory.

.. code:: ipython3

    graphml_name = "LEMMAREL.verbs"
    templates_dir = f"{mydir}/templates"
    rel_macros = [
        ("LEMMAREL1", loadMacro(templates_dir, graphml_name, "LEMMAREL1.verbs")),
        ("LEMMAREL2", loadMacro(templates_dir, graphml_name, "LEMMAREL2.verbs"))
    ]
    settings['separator-line-machine'] = "^</sentence>$"

.. code:: ipython3

    # The objects returned by loadMacro can be plotted:
    plotPatterns(rel_macros[0][1])



.. image:: output_31_0.png


.. code:: ipython3

    reldata = createRel(query, settings, rel_macros, type_name = type_name,
                  fnames = fnameSample, tokenlist = tokenlist, foc_filter = foc.get_item_list())
    reldata.to_csv(f"{output_path}/registers/{type_name}.rel-models.tsv", sep = "\t", index_label = "_model")


.. parsed-literal::

    WARNING: Not provide the temporary path!
    WARNING: Use the default tmp directory: '~/tmp'!
    Building dependency features...



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=60), HTML(value='')))


.. parsed-literal::

    
    Building matrix...
    WARNING: 5 rows have not been found.
    WARNING: 60432 columns have not been found.
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy//destroy.LEMMAREL1.tcmx.bool.pac
    WARNING: Not provide the temporary path!
    WARNING: Use the default tmp directory: '~/tmp'!
    Building dependency features...



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=60), HTML(value='')))


.. parsed-literal::

    
    Building matrix...
    WARNING: 1 rows have not been found.
    WARNING: 60398 columns have not been found.
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy//destroy.LEMMAREL2.tcmx.bool.pac


3.3 Lemmapath
~~~~~~~~~~~~~

Like LEMMAREL, the LEMMAPATH models need 'separator-line-machine' to be
properly set and the templates to be loaded. Unlike LEMMAREL, the
templates are not cumulative: LEMMAPATH1 models only cover those with
one step between target and context word, while LEMMAPATH2 covers those
with two steps. We *could* make them cumulative, but this setup allows
us to give them different weights in PATHweight models.

.. code:: ipython3

    graphml_name = "LEMMAPATH"
    templates_dir = f"{mydir}/templates"
    path_templates = [loadMacro(templates_dir, graphml_name, f"LEMMAPATH{i}") for i in [1, 2, 3]]
    path_macros = [
        # First group includes templates with one and two steps, no weight
        ("LEMMAPATH2", [path_templates[0], path_templates[1]], None),
        # Second group includes templates with up to three steps, no weight
        ("LEMMAPATH3", [path_templates[0], path_templates[1], path_templates[2]], None),
        # Third group includes templates with up to three steps, with weight
        ("LEMMAPATHweight", [path_templates[0], path_templates[1], path_templates[2]], [1, 0.6, 0.3])
    ]
    settings['separator-line-machine'] = "^</sentence>$"

.. code:: ipython3

    pathdata = createPath(query, settings, path_macros, type_name = type_name,
              fnames = fnameSample, tokenlist = tokenlist, foc_filter = foc.get_item_list())
    pathdata.to_csv(f"{output_path}/registers/{type_name}.path-models.tsv", sep = "\t", index_label = "_model")


.. parsed-literal::

    WARNING: Not provide the temporary path!
    WARNING: Use the default tmp directory: '~/tmp'!
    Building dependency features...



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=60), HTML(value='')))


.. parsed-literal::

    
    Building matrix...
    WARNING: Not provide the temporary path!
    WARNING: Use the default tmp directory: '~/tmp'!
    Building dependency features...



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=60), HTML(value='')))


.. parsed-literal::

    
    Building matrix...
    WARNING: 60351 columns have not been found.
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy//destroy.LEMMAPATH2.tcmx.bool.pac
    WARNING: Not provide the temporary path!
    WARNING: Use the default tmp directory: '~/tmp'!
    Building dependency features...



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=60), HTML(value='')))


.. parsed-literal::

    
    Building matrix...
    WARNING: Not provide the temporary path!
    WARNING: Use the default tmp directory: '~/tmp'!
    Building dependency features...



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=60), HTML(value='')))


.. parsed-literal::

    
    Building matrix...
    WARNING: Not provide the temporary path!
    WARNING: Use the default tmp directory: '~/tmp'!
    Building dependency features...



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=60), HTML(value='')))


.. parsed-literal::

    
    Building matrix...
    WARNING: 2 rows have not been found.
    WARNING: 60269 columns have not been found.
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy//destroy.LEMMAPATH3.tcmx.bool.pac
    WARNING: Not provide the temporary path!
    WARNING: Use the default tmp directory: '~/tmp'!
    Building dependency features...



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=60), HTML(value='')))


.. parsed-literal::

    
    Building matrix...
    WARNING: Not provide the temporary path!
    WARNING: Use the default tmp directory: '~/tmp'!
    Building dependency features...



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=60), HTML(value='')))


.. parsed-literal::

    
    Building matrix...
    WARNING: Not provide the temporary path!
    WARNING: Use the default tmp directory: '~/tmp'!
    Building dependency features...



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=60), HTML(value='')))


.. parsed-literal::

    
    Building matrix...
    WARNING: 2 rows have not been found.
    WARNING: 60269 columns have not been found.
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy//destroy.LEMMAPATHweight.tcmx.bool.pac


4 Weight or booleanize
----------------------

Once we have our boolean token-by-feature matrices, we can start
combining them with type-level matrices: first to weight them and then
to obtain second-order features. These functions will require us to
specify the directory where we store our token matrices (in case we want
different directories).

.. code:: ipython3

    token_dir = f"{output_path}/tokens/{type_name}"

4.1. Create/load collocation matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First of all, we need to have a collocation matrix. The following
function checks if the given filename exists and, if it doesn't, it
creates the matrix from scratch.

.. code:: ipython3

    coldir = "/home/projects/semmetrix/NephoSem/input-data/frequency-matrices/QLVLNewsCorpus/"
    freq_fname_CW4 = f"{coldir}/QLVLNews.fullcorpus_CW4.wcmx.freq.pac" # window size of 4

.. code:: ipython3

    #settings['left-span'] = 4
    #settings['right-span = 4']
    freqMTX_CW4 = loadColloc(freq_fname_CW4, settings)
    freqMTX_CW4




.. parsed-literal::

    [4614267, 4614267]             !!!!!!!!!!!!!!!!!!!!!!!!!/num  !!!!!!!!!!!!!!!!!/num  !!!!!!!!!!!!!!!!/num  !!!!!!!!!!!!!!/num  !!!!!!!!!!!!!/name  !!!!!!!!!!!!!/num  !!!!!!!!!!!!/num  ...
    !!!!!!!!!!!!!!!!!!!!!!!!!/num  NaN                            NaN                    NaN                   NaN                 NaN                 NaN                NaN               ...
    !!!!!!!!!!!!!!!!!/num          NaN                            NaN                    NaN                   NaN                 NaN                 NaN                NaN               ...
    !!!!!!!!!!!!!!!!/num           NaN                            NaN                    NaN                   NaN                 NaN                 NaN                NaN               ...
    !!!!!!!!!!!!!!/num             NaN                            NaN                    NaN                   NaN                 NaN                 NaN                NaN               ...
    !!!!!!!!!!!!!/name             NaN                            NaN                    NaN                   NaN                 NaN                 NaN                NaN               ...
    !!!!!!!!!!!!!/num              NaN                            NaN                    NaN                   NaN                 NaN                 NaN                NaN               ...
    !!!!!!!!!!!!/num               NaN                            NaN                    NaN                   NaN                 NaN                 NaN                NaN               ...
    ...                            ...                            ...                    ...                   ...                 ...                 ...                ...               ...



.. code:: ipython3

    freq_fname_CW10 = f"{coldir}/QLVLNews.fullcorpus_CW10.wcmx.freq.pac" # window size of 4
    #settings['left-span'] = 10
    #settings['right-span = 10']
    freqMTX_CW10 = loadColloc(freq_fname_CW10, settings)

4.2 Register PPMI values
~~~~~~~~~~~~~~~~~~~~~~~~

The function below subsets collocation matrices and calculates PMI
values based on collocation matrices and frequencies based on
vocabularies, to register the information in a dataframe. It returns a
specific PPMI dataframe to use for weighting.

.. code:: ipython3

    ppmi = targetPPMI(query.get_item_list(),
               vocabs = {"freq" : full},
              collocs = {"4" : freqMTX_CW4, "10" : freqMTX_CW10},
              type_name = type_name, output_dir = f"{github_dir}/{type_name}/",
              main_matrix = "4")
    ppmi # it returns the PPMI values based on collocs["4"]



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=20120), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1711 sec
    ************************************
    



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=39735), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.2749 sec
    ************************************
    




.. parsed-literal::

    [2, 14453]      !!!/punct  "/punct     '/punct     '40/noun  '44/noun   '67/name   '68_generatie/noun  ...
    verniel/verb    NaN        NaN         NaN         4.903999  4.3824615  NaN        7.2866263           ...
    vernietig/verb  2.2368245  0.20710099  0.25722486  NaN       NaN        4.8763175  NaN                 ...



.. code:: ipython3

    import pandas as pd
    with open(f"{github_dir}/{type_name}/{type_name}.ppmi.tsv", "r") as f:
        pmidf = pd.read_csv(f, sep = "\t", index_col = "cw")
    pmidf




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>pmi_4_verniel/verb</th>
          <th>pmi_4_vernietig/verb</th>
          <th>raw_4_verniel/verb</th>
          <th>raw_4_vernietig/verb</th>
          <th>pmi_10_verniel/verb</th>
          <th>pmi_10_vernietig/verb</th>
          <th>raw_10_verniel/verb</th>
          <th>raw_10_vernietig/verb</th>
          <th>freq</th>
        </tr>
        <tr>
          <th>cw</th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>!!!/punct</th>
          <td>NaN</td>
          <td>2.236824</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>NaN</td>
          <td>1.331356</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>592</td>
        </tr>
        <tr>
          <th>!/name</th>
          <td>-0.776889</td>
          <td>-0.166773</td>
          <td>1.0</td>
          <td>3.0</td>
          <td>-0.575533</td>
          <td>-1.070589</td>
          <td>3.0</td>
          <td>3.0</td>
          <td>19257</td>
        </tr>
        <tr>
          <th>!/punct</th>
          <td>-1.148367</td>
          <td>-0.027426</td>
          <td>5.0</td>
          <td>25.0</td>
          <td>-1.170905</td>
          <td>-0.366678</td>
          <td>12.0</td>
          <td>44.0</td>
          <td>142029</td>
        </tr>
        <tr>
          <th>!?/name</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>3.903481</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>NaN</td>
          <td>73</td>
        </tr>
        <tr>
          <th>"/punct</th>
          <td>-0.416885</td>
          <td>0.207101</td>
          <td>358.0</td>
          <td>1089.0</td>
          <td>-0.523912</td>
          <td>0.023079</td>
          <td>794.0</td>
          <td>2251.0</td>
          <td>4844510</td>
        </tr>
        <tr>
          <th>&amp;/name</th>
          <td>-1.789608</td>
          <td>-1.179493</td>
          <td>1.0</td>
          <td>3.0</td>
          <td>-2.685440</td>
          <td>-1.571058</td>
          <td>1.0</td>
          <td>5.0</td>
          <td>52896</td>
        </tr>
        <tr>
          <th>&amp;Wardwell/name</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>6.980771</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>2</td>
        </tr>
        <tr>
          <th>'/punct</th>
          <td>-0.873048</td>
          <td>0.257225</td>
          <td>128.0</td>
          <td>646.0</td>
          <td>-0.837387</td>
          <td>0.136009</td>
          <td>327.0</td>
          <td>1420.0</td>
          <td>2720890</td>
        </tr>
        <tr>
          <th>'14-'18/num</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>3.126377</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>96</td>
        </tr>
        <tr>
          <th>'33/noun</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>4.272720</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>30</td>
        </tr>
        <tr>
          <th>'39/num</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>6.028908</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>NaN</td>
          <td>9</td>
        </tr>
        <tr>
          <th>'40/noun</th>
          <td>4.903999</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>NaN</td>
          <td>3.994586</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>NaN</td>
          <td>65</td>
        </tr>
        <tr>
          <th>'44/noun</th>
          <td>4.382461</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>NaN</td>
          <td>4.169398</td>
          <td>NaN</td>
          <td>2.0</td>
          <td>NaN</td>
          <td>110</td>
        </tr>
        <tr>
          <th>'65/name</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>4.341713</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>28</td>
        </tr>
        <tr>
          <th>'67/name</th>
          <td>NaN</td>
          <td>4.876317</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>NaN</td>
          <td>3.960346</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>41</td>
        </tr>
        <tr>
          <th>'68_generatie/noun</th>
          <td>7.286626</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>NaN</td>
          <td>6.377214</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>NaN</td>
          <td>6</td>
        </tr>
        <tr>
          <th>'70/num</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.897096</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>NaN</td>
          <td>1447</td>
        </tr>
        <tr>
          <th>'87/noun</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>2.774892</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>NaN</td>
          <td>221</td>
        </tr>
        <tr>
          <th>'89/noun</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>1.730462</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>383</td>
        </tr>
        <tr>
          <th>'90-'91/num</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>5.195999</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>NaN</td>
          <td>20</td>
        </tr>
        <tr>
          <th>'90/num</th>
          <td>NaN</td>
          <td>1.343432</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>0.926176</td>
          <td>1.529732</td>
          <td>1.0</td>
          <td>3.0</td>
          <td>1406</td>
        </tr>
        <tr>
          <th>'91/noun</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>1.954166</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>NaN</td>
          <td>502</td>
        </tr>
        <tr>
          <th>'91/num</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>3.571274</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>61</td>
        </tr>
        <tr>
          <th>'92/noun</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>1.634241</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>NaN</td>
          <td>693</td>
        </tr>
        <tr>
          <th>'92/num</th>
          <td>NaN</td>
          <td>3.847133</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>NaN</td>
          <td>2.935967</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>115</td>
        </tr>
        <tr>
          <th>'94-'96/num</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>5.222913</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>12</td>
        </tr>
        <tr>
          <th>'94/noun</th>
          <td>NaN</td>
          <td>1.685640</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>NaN</td>
          <td>0.772030</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>999</td>
        </tr>
        <tr>
          <th>'95/noun</th>
          <td>NaN</td>
          <td>1.509021</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>NaN</td>
          <td>1.289103</td>
          <td>NaN</td>
          <td>2.0</td>
          <td>1191</td>
        </tr>
        <tr>
          <th>'96/noun</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>1.115897</td>
          <td>NaN</td>
          <td>2.0</td>
          <td>1417</td>
        </tr>
        <tr>
          <th>'98/noun</th>
          <td>NaN</td>
          <td>1.421021</td>
          <td>NaN</td>
          <td>2.0</td>
          <td>NaN</td>
          <td>0.509275</td>
          <td>NaN</td>
          <td>2.0</td>
          <td>2605</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>zwerf_kat/noun</th>
          <td>NaN</td>
          <td>2.551422</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>NaN</td>
          <td>1.657029</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>428</td>
        </tr>
        <tr>
          <th>zwerf_rond/verb</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>1.093625</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>728</td>
        </tr>
        <tr>
          <th>zwerf_route/noun</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>4.646203</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>22</td>
        </tr>
        <tr>
          <th>zwerf_wagen/noun</th>
          <td>4.469473</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>NaN</td>
          <td>3.571332</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>NaN</td>
          <td>101</td>
        </tr>
        <tr>
          <th>zwerm/noun</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>1.950741</td>
          <td>NaN</td>
          <td>2.0</td>
          <td>619</td>
        </tr>
        <tr>
          <th>zwerver/noun</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.959171</td>
          <td>NaN</td>
          <td>2.0</td>
          <td>NaN</td>
          <td>2756</td>
        </tr>
        <tr>
          <th>zwezerik/noun</th>
          <td>NaN</td>
          <td>3.374954</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>NaN</td>
          <td>2.466620</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>184</td>
        </tr>
        <tr>
          <th>zwicht/verb</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>1.237893</td>
          <td>NaN</td>
          <td>2.0</td>
          <td>NaN</td>
          <td>2090</td>
        </tr>
        <tr>
          <th>zwijg/adj</th>
          <td>1.408599</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>NaN</td>
          <td>0.505472</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>NaN</td>
          <td>2152</td>
        </tr>
        <tr>
          <th>zwijg/verb</th>
          <td>NaN</td>
          <td>-0.983139</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>-0.701847</td>
          <td>-0.791438</td>
          <td>2.0</td>
          <td>3.0</td>
          <td>14482</td>
        </tr>
        <tr>
          <th>zwijg_dood/verb</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>1.840769</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>NaN</td>
          <td>563</td>
        </tr>
        <tr>
          <th>zwijg_geld/noun</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>2.556523</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>170</td>
        </tr>
        <tr>
          <th>zwijg_stil/adj</th>
          <td>NaN</td>
          <td>1.571040</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>NaN</td>
          <td>0.661037</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>1120</td>
        </tr>
        <tr>
          <th>zwijn/noun</th>
          <td>2.080561</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>NaN</td>
          <td>1.876387</td>
          <td>0.688183</td>
          <td>2.0</td>
          <td>1.0</td>
          <td>1104</td>
        </tr>
        <tr>
          <th>zyn/noun</th>
          <td>NaN</td>
          <td>4.658064</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>NaN</td>
          <td>3.747006</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>51</td>
        </tr>
        <tr>
          <th>140/num</th>
          <td>NaN</td>
          <td>4.962221</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>NaN</td>
          <td>4.064352</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>38</td>
        </tr>
        <tr>
          <th>iroki/noun</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>7.475827</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>NaN</td>
          <td>2</td>
        </tr>
        <tr>
          <th>©$^NRC/noun</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>4.873137</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>NaN</td>
          <td>50</td>
        </tr>
        <tr>
          <th>©/noun</th>
          <td>1.242708</td>
          <td>0.061064</td>
          <td>2.0</td>
          <td>1.0</td>
          <td>0.455100</td>
          <td>0.653191</td>
          <td>2.0</td>
          <td>4.0</td>
          <td>5868</td>
        </tr>
        <tr>
          <th>Öcalan/name</th>
          <td>NaN</td>
          <td>0.775288</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>NaN</td>
          <td>-0.125323</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>2497</td>
        </tr>
        <tr>
          <th>à/vg</th>
          <td>NaN</td>
          <td>-1.945505</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>-1.260012</td>
          <td>-0.907770</td>
          <td>3.0</td>
          <td>7.0</td>
          <td>37763</td>
        </tr>
        <tr>
          <th>á/name</th>
          <td>5.825108</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>NaN</td>
          <td>4.924430</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>NaN</td>
          <td>26</td>
        </tr>
        <tr>
          <th>álle/adj</th>
          <td>2.752684</td>
          <td>2.264187</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.849285</td>
          <td>1.354229</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>560</td>
        </tr>
        <tr>
          <th>één/adv</th>
          <td>NaN</td>
          <td>1.423189</td>
          <td>NaN</td>
          <td>2.0</td>
          <td>1.006731</td>
          <td>0.511675</td>
          <td>2.0</td>
          <td>2.0</td>
          <td>2595</td>
        </tr>
        <tr>
          <th>één/fixed</th>
          <td>0.351682</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>NaN</td>
          <td>-0.549003</td>
          <td>-1.044059</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>6184</td>
        </tr>
        <tr>
          <th>één/noun</th>
          <td>NaN</td>
          <td>0.095581</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>0.375521</td>
          <td>-0.812682</td>
          <td>2.0</td>
          <td>1.0</td>
          <td>4905</td>
        </tr>
        <tr>
          <th>één/num</th>
          <td>0.436258</td>
          <td>-0.086141</td>
          <td>60.0</td>
          <td>58.0</td>
          <td>0.367265</td>
          <td>0.001201</td>
          <td>138.0</td>
          <td>157.0</td>
          <td>341260</td>
        </tr>
        <tr>
          <th>één/pron</th>
          <td>-0.165669</td>
          <td>-0.607645</td>
          <td>21.0</td>
          <td>22.0</td>
          <td>-0.088517</td>
          <td>-0.565874</td>
          <td>56.0</td>
          <td>57.0</td>
          <td>218147</td>
        </tr>
        <tr>
          <th>être/name</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>4.102514</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>NaN</td>
          <td>59</td>
        </tr>
        <tr>
          <th>überhaupt/adv</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>-0.102450</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>2400</td>
        </tr>
      </tbody>
    </table>
    <p>32343 rows × 9 columns</p>
    </div>



4.3 Implement weighting on selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This step is performed on all the matrices created up to this moment. A
useful thing to do first is to combine all the first-order register
information we have from the different kinds of models.

.. code:: ipython3

    registers = loadFocRegisters(f"{output_path}/registers", type_name)

Once we have the registers, we can also set the values for our ``PPMI``
setting with the ``weighting`` dictionary. A value ``None`` indicates
that no weighting is applied, while matrices as values (a boolean
version for selection instead of weighting, for example) are used to
weight the tokens.

.. code:: ipython3

    from semasioFlow.utils import booleanize
    weighting = {
        "no" : None,
        "selection" : booleanize(ppmi, include_negative=False),
        "weight" : ppmi
    }

.. code:: ipython3

    x = "weight"

.. code:: ipython3

    weighting[x] if weighting[x] else "no matrix"




.. parsed-literal::

    [2, 14453]      !!!/punct  "/punct     '/punct     '40/noun  '44/noun   '67/name   '68_generatie/noun  ...
    verniel/verb    NaN        NaN         NaN         4.903999  4.3824615  NaN        7.2866263           ...
    vernietig/verb  2.2368245  0.20710099  0.25722486  NaN       NaN        4.8763175  NaN                 ...



.. code:: ipython3

    weight_data = weightTokens(token_dir, weighting, registers)


.. parsed-literal::

    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10all.PPMIno.tcmx.weight.pac
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10all.PPMIselection.tcmx.weight.pac
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10all.PPMIweight.tcmx.weight.pac
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10lex.PPMIno.tcmx.weight.pac
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10lex.PPMIselection.tcmx.weight.pac
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10lex.PPMIweight.tcmx.weight.pac
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3all.PPMIno.tcmx.weight.pac
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3all.PPMIselection.tcmx.weight.pac
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3all.PPMIweight.tcmx.weight.pac
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3lex.PPMIno.tcmx.weight.pac
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3lex.PPMIselection.tcmx.weight.pac
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3lex.PPMIweight.tcmx.weight.pac
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5all.PPMIno.tcmx.weight.pac
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5all.PPMIselection.tcmx.weight.pac
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5all.PPMIweight.tcmx.weight.pac
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5lex.PPMIno.tcmx.weight.pac
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5lex.PPMIselection.tcmx.weight.pac
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5lex.PPMIweight.tcmx.weight.pac
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10all.PPMIno.tcmx.weight.pac
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10all.PPMIselection.tcmx.weight.pac
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10all.PPMIweight.tcmx.weight.pac
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10lex.PPMIno.tcmx.weight.pac
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10lex.PPMIselection.tcmx.weight.pac
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10lex.PPMIweight.tcmx.weight.pac
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3all.PPMIno.tcmx.weight.pac
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3all.PPMIselection.tcmx.weight.pac
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3all.PPMIweight.tcmx.weight.pac
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3lex.PPMIno.tcmx.weight.pac
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3lex.PPMIselection.tcmx.weight.pac
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3lex.PPMIweight.tcmx.weight.pac
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5all.PPMIno.tcmx.weight.pac
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5all.PPMIselection.tcmx.weight.pac
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5all.PPMIweight.tcmx.weight.pac
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5lex.PPMIno.tcmx.weight.pac
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5lex.PPMIselection.tcmx.weight.pac
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5lex.PPMIweight.tcmx.weight.pac
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL1.PPMIno.tcmx.weight.pac
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL1.PPMIselection.tcmx.weight.pac
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL1.PPMIweight.tcmx.weight.pac
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL2.PPMIno.tcmx.weight.pac
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL2.PPMIselection.tcmx.weight.pac
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL2.PPMIweight.tcmx.weight.pac
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH2.PPMIno.tcmx.weight.pac
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH2.PPMIselection.tcmx.weight.pac
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH2.PPMIweight.tcmx.weight.pac
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH3.PPMIno.tcmx.weight.pac
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH3.PPMIselection.tcmx.weight.pac
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH3.PPMIweight.tcmx.weight.pac
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATHweight.PPMIno.tcmx.weight.pac
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATHweight.PPMIselection.tcmx.weight.pac
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATHweight.PPMIweight.tcmx.weight.pac


.. code:: ipython3

    # new model register
    weight_data["model_register"]
    # weight_data["model_register"].to_csv(f"{output_path}/register/destroy.focmodels.tsv", sep = '\t', index_label = "_model")




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>LEMMAPATH</th>
          <th>LEMMAREL</th>
          <th>bound</th>
          <th>foc_base</th>
          <th>foc_context_words</th>
          <th>foc_pmi</th>
          <th>foc_pos</th>
          <th>foc_win</th>
          <th>tokens</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>destroy.LEMMAPATH2.PPMIno</th>
          <td>LEMMAPATH2</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>LEMMAPATH</td>
          <td>172</td>
          <td>no</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.LEMMAPATH2.PPMIselection</th>
          <td>LEMMAPATH2</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>LEMMAPATH</td>
          <td>132</td>
          <td>selection</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.LEMMAPATH2.PPMIweight</th>
          <td>LEMMAPATH2</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>LEMMAPATH</td>
          <td>132</td>
          <td>weight</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.LEMMAPATH3.PPMIno</th>
          <td>LEMMAPATH3</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>LEMMAPATH</td>
          <td>250</td>
          <td>no</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>58</td>
        </tr>
        <tr>
          <th>destroy.LEMMAPATH3.PPMIselection</th>
          <td>LEMMAPATH3</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>LEMMAPATH</td>
          <td>184</td>
          <td>selection</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>58</td>
        </tr>
        <tr>
          <th>destroy.LEMMAPATH3.PPMIweight</th>
          <td>LEMMAPATH3</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>LEMMAPATH</td>
          <td>184</td>
          <td>weight</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>58</td>
        </tr>
        <tr>
          <th>destroy.LEMMAPATHweight.PPMIno</th>
          <td>LEMMAPATHweight</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>LEMMAPATH</td>
          <td>250</td>
          <td>no</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>58</td>
        </tr>
        <tr>
          <th>destroy.LEMMAPATHweight.PPMIselection</th>
          <td>LEMMAPATHweight</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>LEMMAPATH</td>
          <td>184</td>
          <td>selection</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>58</td>
        </tr>
        <tr>
          <th>destroy.LEMMAPATHweight.PPMIweight</th>
          <td>LEMMAPATHweight</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>LEMMAPATH</td>
          <td>184</td>
          <td>weight</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>58</td>
        </tr>
        <tr>
          <th>destroy.LEMMAREL1.PPMIno</th>
          <td>NaN</td>
          <td>LEMMAREL1</td>
          <td>NaN</td>
          <td>LEMMAREL</td>
          <td>97</td>
          <td>no</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>55</td>
        </tr>
        <tr>
          <th>destroy.LEMMAREL1.PPMIselection</th>
          <td>NaN</td>
          <td>LEMMAREL1</td>
          <td>NaN</td>
          <td>LEMMAREL</td>
          <td>72</td>
          <td>selection</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>50</td>
        </tr>
        <tr>
          <th>destroy.LEMMAREL1.PPMIweight</th>
          <td>NaN</td>
          <td>LEMMAREL1</td>
          <td>NaN</td>
          <td>LEMMAREL</td>
          <td>72</td>
          <td>weight</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>50</td>
        </tr>
        <tr>
          <th>destroy.LEMMAREL2.PPMIno</th>
          <td>NaN</td>
          <td>LEMMAREL2</td>
          <td>NaN</td>
          <td>LEMMAREL</td>
          <td>128</td>
          <td>no</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>59</td>
        </tr>
        <tr>
          <th>destroy.LEMMAREL2.PPMIselection</th>
          <td>NaN</td>
          <td>LEMMAREL2</td>
          <td>NaN</td>
          <td>LEMMAREL</td>
          <td>97</td>
          <td>selection</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>58</td>
        </tr>
        <tr>
          <th>destroy.LEMMAREL2.PPMIweight</th>
          <td>NaN</td>
          <td>LEMMAREL2</td>
          <td>NaN</td>
          <td>LEMMAREL</td>
          <td>97</td>
          <td>weight</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>58</td>
        </tr>
        <tr>
          <th>destroy.bound10-10all.PPMIno</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>True</td>
          <td>BOW</td>
          <td>354</td>
          <td>no</td>
          <td>all</td>
          <td>10-10</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.bound10-10all.PPMIselection</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>True</td>
          <td>BOW</td>
          <td>247</td>
          <td>selection</td>
          <td>all</td>
          <td>10-10</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.bound10-10all.PPMIweight</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>True</td>
          <td>BOW</td>
          <td>247</td>
          <td>weight</td>
          <td>all</td>
          <td>10-10</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.bound10-10lex.PPMIno</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>True</td>
          <td>BOW</td>
          <td>249</td>
          <td>no</td>
          <td>lex</td>
          <td>10-10</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.bound10-10lex.PPMIselection</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>True</td>
          <td>BOW</td>
          <td>179</td>
          <td>selection</td>
          <td>lex</td>
          <td>10-10</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.bound10-10lex.PPMIweight</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>True</td>
          <td>BOW</td>
          <td>179</td>
          <td>weight</td>
          <td>lex</td>
          <td>10-10</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.bound3-3all.PPMIno</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>True</td>
          <td>BOW</td>
          <td>146</td>
          <td>no</td>
          <td>all</td>
          <td>3-3</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.bound3-3all.PPMIselection</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>True</td>
          <td>BOW</td>
          <td>118</td>
          <td>selection</td>
          <td>all</td>
          <td>3-3</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.bound3-3all.PPMIweight</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>True</td>
          <td>BOW</td>
          <td>118</td>
          <td>weight</td>
          <td>all</td>
          <td>3-3</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.bound3-3lex.PPMIno</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>True</td>
          <td>BOW</td>
          <td>100</td>
          <td>no</td>
          <td>lex</td>
          <td>3-3</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.bound3-3lex.PPMIselection</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>True</td>
          <td>BOW</td>
          <td>86</td>
          <td>selection</td>
          <td>lex</td>
          <td>3-3</td>
          <td>55</td>
        </tr>
        <tr>
          <th>destroy.bound3-3lex.PPMIweight</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>True</td>
          <td>BOW</td>
          <td>86</td>
          <td>weight</td>
          <td>lex</td>
          <td>3-3</td>
          <td>55</td>
        </tr>
        <tr>
          <th>destroy.bound5-5all.PPMIno</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>True</td>
          <td>BOW</td>
          <td>224</td>
          <td>no</td>
          <td>all</td>
          <td>5-5</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.bound5-5all.PPMIselection</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>True</td>
          <td>BOW</td>
          <td>169</td>
          <td>selection</td>
          <td>all</td>
          <td>5-5</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.bound5-5all.PPMIweight</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>True</td>
          <td>BOW</td>
          <td>169</td>
          <td>weight</td>
          <td>all</td>
          <td>5-5</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.bound5-5lex.PPMIno</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>True</td>
          <td>BOW</td>
          <td>156</td>
          <td>no</td>
          <td>lex</td>
          <td>5-5</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.bound5-5lex.PPMIselection</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>True</td>
          <td>BOW</td>
          <td>125</td>
          <td>selection</td>
          <td>lex</td>
          <td>5-5</td>
          <td>59</td>
        </tr>
        <tr>
          <th>destroy.bound5-5lex.PPMIweight</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>True</td>
          <td>BOW</td>
          <td>125</td>
          <td>weight</td>
          <td>lex</td>
          <td>5-5</td>
          <td>59</td>
        </tr>
        <tr>
          <th>destroy.nobound10-10all.PPMIno</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>False</td>
          <td>BOW</td>
          <td>500</td>
          <td>no</td>
          <td>all</td>
          <td>10-10</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.nobound10-10all.PPMIselection</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>False</td>
          <td>BOW</td>
          <td>333</td>
          <td>selection</td>
          <td>all</td>
          <td>10-10</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.nobound10-10all.PPMIweight</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>False</td>
          <td>BOW</td>
          <td>333</td>
          <td>weight</td>
          <td>all</td>
          <td>10-10</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.nobound10-10lex.PPMIno</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>False</td>
          <td>BOW</td>
          <td>349</td>
          <td>no</td>
          <td>lex</td>
          <td>10-10</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.nobound10-10lex.PPMIselection</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>False</td>
          <td>BOW</td>
          <td>240</td>
          <td>selection</td>
          <td>lex</td>
          <td>10-10</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.nobound10-10lex.PPMIweight</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>False</td>
          <td>BOW</td>
          <td>240</td>
          <td>weight</td>
          <td>lex</td>
          <td>10-10</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.nobound3-3all.PPMIno</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>False</td>
          <td>BOW</td>
          <td>178</td>
          <td>no</td>
          <td>all</td>
          <td>3-3</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.nobound3-3all.PPMIselection</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>False</td>
          <td>BOW</td>
          <td>144</td>
          <td>selection</td>
          <td>all</td>
          <td>3-3</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.nobound3-3all.PPMIweight</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>False</td>
          <td>BOW</td>
          <td>144</td>
          <td>weight</td>
          <td>all</td>
          <td>3-3</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.nobound3-3lex.PPMIno</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>False</td>
          <td>BOW</td>
          <td>115</td>
          <td>no</td>
          <td>lex</td>
          <td>3-3</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.nobound3-3lex.PPMIselection</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>False</td>
          <td>BOW</td>
          <td>96</td>
          <td>selection</td>
          <td>lex</td>
          <td>3-3</td>
          <td>55</td>
        </tr>
        <tr>
          <th>destroy.nobound3-3lex.PPMIweight</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>False</td>
          <td>BOW</td>
          <td>96</td>
          <td>weight</td>
          <td>lex</td>
          <td>3-3</td>
          <td>55</td>
        </tr>
        <tr>
          <th>destroy.nobound5-5all.PPMIno</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>False</td>
          <td>BOW</td>
          <td>283</td>
          <td>no</td>
          <td>all</td>
          <td>5-5</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.nobound5-5all.PPMIselection</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>False</td>
          <td>BOW</td>
          <td>215</td>
          <td>selection</td>
          <td>all</td>
          <td>5-5</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.nobound5-5all.PPMIweight</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>False</td>
          <td>BOW</td>
          <td>215</td>
          <td>weight</td>
          <td>all</td>
          <td>5-5</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.nobound5-5lex.PPMIno</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>False</td>
          <td>BOW</td>
          <td>194</td>
          <td>no</td>
          <td>lex</td>
          <td>5-5</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.nobound5-5lex.PPMIselection</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>False</td>
          <td>BOW</td>
          <td>153</td>
          <td>selection</td>
          <td>lex</td>
          <td>5-5</td>
          <td>59</td>
        </tr>
        <tr>
          <th>destroy.nobound5-5lex.PPMIweight</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>False</td>
          <td>BOW</td>
          <td>153</td>
          <td>weight</td>
          <td>lex</td>
          <td>5-5</td>
          <td>59</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    # token_level register
    weight_data["token_register"]
    github_type = f"{github_dir}/{type_name}"
    if not os.path.exists(github_type):
        os.makedirs(github_type)
        
    weight_data["token_register"].to_csv(f"{github_type}/{type_name}.variables.tsv", sep = '\t', index_label = "_id")

5 Second-order dimensions
~~~~~~~~~~~~~~~~~~~~~~~~~

The final step to obtain our token-level vectors is to multiply the
token-foc matrices for type-level matrices to obtain second-order
vectors. We will loop over the models in the index of
``weight_data["model_register"]`` and over second-order parameter
settings to filter ``freqMTX_CW4`` and obtain different models.

.. code:: ipython3

    soc_pos = {
        "all" : foc,
        "nav" : soc
    }
    lengths = ["FOC", 5000] # a number will take the most frequent; something else will take the FOC items

.. code:: ipython3

    socdata = createSoc(token_dir,
                        registers = weight_data['model_register'],
                       soc_pos = soc_pos, lengths = lengths,
                       socMTX = freqMTX_CW4, store_focdists = f"{output_path}/cws/{type_name}/")



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=24093), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1729 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.001491 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH2.PPMIno.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=655500), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 2.117 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.007475 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH2.PPMIno.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=12043), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1125 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.0008636 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH2.PPMIno.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=607722), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.86 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.007209 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH2.PPMIno.LENGTH5000.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=13548), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1184 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.0007915 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH2.PPMIselection.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=480697), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.499 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.008267 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH2.PPMIselection.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=7465), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.0984 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.000854 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH2.PPMIselection.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=440772), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.38 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.005088 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH2.PPMIselection.LENGTH5000.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=13548), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1145 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.0007269 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH2.PPMIweight.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=480697), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.515 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.005095 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH2.PPMIweight.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=7465), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.09723 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.0008435 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH2.PPMIweight.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=440772), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.417 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.005049 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH2.PPMIweight.LENGTH5000.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=49751), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.2302 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.001689 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH3.PPMIno.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=944967), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 2.878 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.01209 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH3.PPMIno.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=26418), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1561 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.001496 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH3.PPMIno.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=868679), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 2.627 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.01213 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH3.PPMIno.LENGTH5000.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=25955), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1547 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.001198 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH3.PPMIselection.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=669228), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 2.038 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.009572 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH3.PPMIselection.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=14679), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1352 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.001088 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH3.PPMIselection.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=608483), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.937 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.007526 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH3.PPMIselection.LENGTH5000.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=25955), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1575 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.001067 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH3.PPMIweight.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=669228), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 2.059 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.007737 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH3.PPMIweight.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=14679), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1219 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.001002 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH3.PPMIweight.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=608483), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.923 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.007514 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH3.PPMIweight.LENGTH5000.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=49751), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.2264 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.001769 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATHweight.PPMIno.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=944967), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 2.836 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.01203 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATHweight.PPMIno.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=26418), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1581 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.001433 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATHweight.PPMIno.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=868679), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 2.629 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.01201 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATHweight.PPMIno.LENGTH5000.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=25955), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.155 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.001114 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATHweight.PPMIselection.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=669228), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 2.021 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.007819 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATHweight.PPMIselection.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=14679), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1193 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.000993 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATHweight.PPMIselection.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=608483), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.874 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.007537 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATHweight.PPMIselection.LENGTH5000.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=25955), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1568 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.001076 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATHweight.PPMIweight.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=669228), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 2.117 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.00793 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATHweight.PPMIweight.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=14679), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1243 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.001037 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATHweight.PPMIweight.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=608483), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.905 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.00774 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATHweight.PPMIweight.LENGTH5000.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=7084), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.09658 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.0008082 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL1.PPMIno.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=345969), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.125 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.005366 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL1.PPMIno.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=3382), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.09702 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.0008752 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL1.PPMIno.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=316205), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.049 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.00369 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL1.PPMIno.LENGTH5000.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=3444), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1143 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.0007906 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL1.PPMIselection.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=233420), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.7935 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.002531 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL1.PPMIselection.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=1912), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.08076 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.001036 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL1.PPMIselection.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=208127), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.7122 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.002427 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL1.PPMIselection.LENGTH5000.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=3444), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.103 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.0008297 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL1.PPMIweight.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=233420), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.7789 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.002471 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL1.PPMIweight.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=1912), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.08079 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.000875 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL1.PPMIweight.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=208127), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.7331 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.002471 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL1.PPMIweight.LENGTH5000.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=12575), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1129 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.0008404 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL2.PPMIno.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=466059), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.464 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.005347 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL2.PPMIno.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=7070), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.09652 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.0007143 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL2.PPMIno.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=425916), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.345 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.004726 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL2.PPMIno.LENGTH5000.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=6795), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.0968 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.0006914 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL2.PPMIselection.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=334765), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.083 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.003463 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL2.PPMIselection.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=4352), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.08875 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.0007186 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL2.PPMIselection.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=301143), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.9848 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.003653 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL2.PPMIselection.LENGTH5000.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=6795), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.0959 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.0006659 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL2.PPMIweight.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=334765), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.077 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.003438 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL2.PPMIweight.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=4352), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.0875 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.00067 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL2.PPMIweight.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=301143), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.9803 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.003713 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL2.PPMIweight.LENGTH5000.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=96085), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.3931 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.003366 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10all.PPMIno.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=1308688), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 3.92 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.02222 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10all.PPMIno.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=50349), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.2328 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.002342 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10all.PPMIno.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=1200082), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 3.701 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.0222 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10all.PPMIno.LENGTH5000.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=45442), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.2178 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.001664 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10all.PPMIselection.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=888273), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 2.713 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.01472 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10all.PPMIselection.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=24818), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1877 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.001389 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10all.PPMIselection.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=803057), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 2.423 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.01226 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10all.PPMIselection.LENGTH5000.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=45442), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.2127 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.001605 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10all.PPMIweight.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=888273), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 2.734 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.01243 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10all.PPMIweight.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=24818), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1535 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.001403 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10all.PPMIweight.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=803057), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 2.49 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.01213 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10all.PPMIweight.LENGTH5000.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=45467), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.2191 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.001604 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10lex.PPMIno.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=901848), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 2.789 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.0119 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10lex.PPMIno.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=34934), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1829 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.001472 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10lex.PPMIno.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=815024), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 2.43 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.01168 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10lex.PPMIno.LENGTH5000.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=23177), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1458 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.001024 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10lex.PPMIselection.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=633938), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.917 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.007821 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10lex.PPMIselection.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=17884), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.132 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.0009747 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10lex.PPMIselection.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=564913), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.753 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.007611 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10lex.PPMIselection.LENGTH5000.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=23177), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1464 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.001046 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10lex.PPMIweight.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=633938), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.968 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.007946 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10lex.PPMIweight.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=17884), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1322 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.0009971 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10lex.PPMIweight.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=564913), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.836 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.008718 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10lex.PPMIweight.LENGTH5000.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=17994), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1714 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.0008237 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3all.PPMIno.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=571633), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.803 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.005889 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3all.PPMIno.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=9668), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1052 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.0007412 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3all.PPMIno.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=531719), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.67 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.006024 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3all.PPMIno.LENGTH5000.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=10958), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1084 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.000772 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3all.PPMIselection.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=434712), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.365 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.004593 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3all.PPMIselection.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=6236), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.09364 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.0007091 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3all.PPMIselection.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=397269), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.261 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.004466 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3all.PPMIselection.LENGTH5000.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=10958), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1097 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.0007455 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3all.PPMIweight.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=434712), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.41 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.004712 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3all.PPMIweight.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=6236), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.09539 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.0007162 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3all.PPMIweight.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=397269), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.268 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.004472 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3all.PPMIweight.LENGTH5000.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=7890), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.0997 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.000694 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3lex.PPMIno.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=371789), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.166 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.003595 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3lex.PPMIno.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=6437), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.09527 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.000694 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3lex.PPMIno.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=337463), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.084 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.003632 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3lex.PPMIno.LENGTH5000.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=5462), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.09439 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.0008645 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3lex.PPMIselection.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=303633), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.018 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.003407 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3lex.PPMIselection.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=4438), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.08937 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.0008781 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3lex.PPMIselection.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=271062), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.905 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.003272 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3lex.PPMIselection.LENGTH5000.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=5462), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.09149 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.0007188 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3lex.PPMIweight.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=303633), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.019 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.00324 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3lex.PPMIweight.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=4438), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.08963 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.000561 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3lex.PPMIweight.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=271062), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.9081 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.003158 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3lex.PPMIweight.LENGTH5000.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=41261), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.2126 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.001478 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5all.PPMIno.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=866454), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 2.691 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.01108 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5all.PPMIno.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=22023), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.147 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.001284 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5all.PPMIno.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=804478), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 2.504 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.01163 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5all.PPMIno.LENGTH5000.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=21980), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1863 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.001029 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5all.PPMIselection.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=617502), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.92 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.007219 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5all.PPMIselection.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=12517), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1146 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.0008993 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5all.PPMIselection.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=564341), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.756 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.006953 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5all.PPMIselection.LENGTH5000.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=21980), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1433 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.000978 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5all.PPMIweight.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=617502), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.963 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.007147 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5all.PPMIweight.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=12517), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1139 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.0008738 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5all.PPMIweight.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=564341), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.765 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.006868 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5all.PPMIweight.LENGTH5000.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=18866), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1357 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.0008879 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5lex.PPMIno.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=580877), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.765 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.006382 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5lex.PPMIno.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=14996), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1201 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.0008733 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5lex.PPMIno.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=529751), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.652 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.006263 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5lex.PPMIno.LENGTH5000.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=11382), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1095 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.0007095 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5lex.PPMIselection.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=443196), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.394 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.004869 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5lex.PPMIselection.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=9096), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1022 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.0006917 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5lex.PPMIselection.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=397751), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.26 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.004791 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5lex.PPMIselection.LENGTH5000.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=11382), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1141 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.0007432 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5lex.PPMIweight.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=443196), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.397 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.004876 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5lex.PPMIweight.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=9096), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1023 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.0007215 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5lex.PPMIweight.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=397751), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.293 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.004869 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5lex.PPMIweight.LENGTH5000.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=181198), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.629 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.005537 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10all.PPMIno.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=1780457), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 5.463 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.04231 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10all.PPMIno.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=94573), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.4048 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.004593 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10all.PPMIno.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=1618725), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 5.001 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.03855 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10all.PPMIno.LENGTH5000.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=78577), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.3183 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.002484 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10all.PPMIselection.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=1156629), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 3.58 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.02118 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10all.PPMIselection.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=41851), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.2047 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.002087 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10all.PPMIselection.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=1038086), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 3.397 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.02062 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10all.PPMIselection.LENGTH5000.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=78577), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.3358 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.002592 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10all.PPMIweight.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=1156629), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 3.698 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.02096 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10all.PPMIweight.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=41851), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.2064 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.002146 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10all.PPMIweight.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=1038086), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 3.137 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.01999 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10all.PPMIweight.LENGTH5000.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=83816), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.3335 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.00271 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10lex.PPMIno.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=1213027), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 3.691 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.02261 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10lex.PPMIno.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=64835), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.3234 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.002644 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10lex.PPMIno.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=1089264), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 3.387 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.01923 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10lex.PPMIno.LENGTH5000.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=39203), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1957 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.001496 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10lex.PPMIselection.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=814295), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 2.552 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.0109 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10lex.PPMIselection.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=29781), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1729 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.001354 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10lex.PPMIselection.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=720473), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 2.302 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.01079 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10lex.PPMIselection.LENGTH5000.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=39203), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.2031 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.001546 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10lex.PPMIweight.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=814295), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 2.617 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.02885 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10lex.PPMIweight.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=29781), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.2135 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.001433 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10lex.PPMIweight.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=720473), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 2.262 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.0127 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10lex.PPMIweight.LENGTH5000.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=26435), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1603 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.001012 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3all.PPMIno.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=693714), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 2.163 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.007782 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3all.PPMIno.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=13558), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1163 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.000947 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3all.PPMIno.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=645841), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 2.007 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.007876 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3all.PPMIno.LENGTH5000.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=16147), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1654 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.0008163 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3all.PPMIselection.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=529547), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.677 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.005461 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3all.PPMIselection.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=8356), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1002 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.0009284 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3all.PPMIselection.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=485673), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.599 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.005555 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3all.PPMIselection.LENGTH5000.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=16147), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1251 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.0007937 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3all.PPMIweight.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=529547), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.671 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.005523 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3all.PPMIweight.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=8356), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.09961 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.0009134 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3all.PPMIweight.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=485673), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.525 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.005466 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3all.PPMIweight.LENGTH5000.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=10340), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1098 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.0007041 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3lex.PPMIno.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=426693), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.35 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.004709 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3lex.PPMIno.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=8544), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1014 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.0006781 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3lex.PPMIno.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=388296), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.241 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.00497 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3lex.PPMIno.LENGTH5000.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=6633), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1134 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.0006349 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3lex.PPMIselection.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=335101), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.063 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.00326 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3lex.PPMIselection.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=5410), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1141 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.0006046 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3lex.PPMIselection.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=299172), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.9726 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.003488 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3lex.PPMIselection.LENGTH5000.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=6633), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1119 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.0006382 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3lex.PPMIweight.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=335101), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.096 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.003359 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3lex.PPMIweight.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=5410), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.09223 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.0008216 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3lex.PPMIweight.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=299172), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.9863 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.00344 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3lex.PPMIweight.LENGTH5000.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=63967), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.2746 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.001973 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5all.PPMIno.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=1072922), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 3.252 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.01631 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5all.PPMIno.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=34003), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1874 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.00177 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5all.PPMIno.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=991017), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 2.999 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.01609 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5all.PPMIno.LENGTH5000.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=34396), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.2204 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.001408 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5all.PPMIselection.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=768983), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 2.343 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.01646 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5all.PPMIselection.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=18813), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1349 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.001114 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5all.PPMIselection.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=699449), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 2.109 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.01008 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5all.PPMIselection.LENGTH5000.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=34396), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1818 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.001461 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5all.PPMIweight.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=768983), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 2.308 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.01162 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5all.PPMIweight.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=18813), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1926 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.001259 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5all.PPMIweight.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=699449), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 2.215 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.009971 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5all.PPMIweight.LENGTH5000.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=28247), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1643 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.001181 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5lex.PPMIno.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=707808), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 2.215 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.008306 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5lex.PPMIno.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=22764), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1484 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.001104 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5lex.PPMIno.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=643194), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 2.005 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.01078 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5lex.PPMIno.LENGTH5000.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=16350), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1254 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.0008755 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5lex.PPMIselection.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=528712), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.741 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.006452 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5lex.PPMIselection.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=13093), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1196 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.000947 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5lex.PPMIselection.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=471982), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.563 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.006557 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5lex.PPMIselection.LENGTH5000.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=16350), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1487 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.0009754 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5lex.PPMIweight.LENGTHFOC.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=528712), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.718 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.006962 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5lex.PPMIweight.LENGTH5000.SOCPOSall.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=13093), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 0.1438 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.0008528 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5lex.PPMIweight.LENGTHFOC.SOCPOSnav.tcmx.soc.pac



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=471982), HTML(value='')))


.. parsed-literal::

    
    
    ************************************
    function    = compute_association
      time      = 1.565 sec
    ************************************
    
    
    ************************************
    function    = compute_distance
      time      = 0.006367 sec
    ************************************
    
      Operation: addition 'token-feature weight matrix' X 'socc matrix'...
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5lex.PPMIweight.LENGTH5000.SOCPOSnav.tcmx.soc.pac


.. code:: ipython3

    socdata




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>LEMMAPATH</th>
          <th>LEMMAREL</th>
          <th>bound</th>
          <th>foc_base</th>
          <th>foc_context_words</th>
          <th>foc_pmi</th>
          <th>foc_pos</th>
          <th>foc_win</th>
          <th>soc_length</th>
          <th>soc_pos</th>
          <th>tokens</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>destroy.LEMMAPATH2.PPMIno.LENGTH5000.SOCPOSall</th>
          <td>LEMMAPATH2</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>LEMMAPATH</td>
          <td>172</td>
          <td>no</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>5000</td>
          <td>all</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.LEMMAPATH2.PPMIno.LENGTH5000.SOCPOSnav</th>
          <td>LEMMAPATH2</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>LEMMAPATH</td>
          <td>172</td>
          <td>no</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>5000</td>
          <td>nav</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.LEMMAPATH2.PPMIno.LENGTHFOC.SOCPOSall</th>
          <td>LEMMAPATH2</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>LEMMAPATH</td>
          <td>172</td>
          <td>no</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>FOC</td>
          <td>all</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.LEMMAPATH2.PPMIno.LENGTHFOC.SOCPOSnav</th>
          <td>LEMMAPATH2</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>LEMMAPATH</td>
          <td>172</td>
          <td>no</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>FOC</td>
          <td>nav</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.LEMMAPATH2.PPMIselection.LENGTH5000.SOCPOSall</th>
          <td>LEMMAPATH2</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>LEMMAPATH</td>
          <td>132</td>
          <td>selection</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>5000</td>
          <td>all</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.LEMMAPATH2.PPMIselection.LENGTH5000.SOCPOSnav</th>
          <td>LEMMAPATH2</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>LEMMAPATH</td>
          <td>132</td>
          <td>selection</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>5000</td>
          <td>nav</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.LEMMAPATH2.PPMIselection.LENGTHFOC.SOCPOSall</th>
          <td>LEMMAPATH2</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>LEMMAPATH</td>
          <td>132</td>
          <td>selection</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>FOC</td>
          <td>all</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.LEMMAPATH2.PPMIselection.LENGTHFOC.SOCPOSnav</th>
          <td>LEMMAPATH2</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>LEMMAPATH</td>
          <td>132</td>
          <td>selection</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>FOC</td>
          <td>nav</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.LEMMAPATH2.PPMIweight.LENGTH5000.SOCPOSall</th>
          <td>LEMMAPATH2</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>LEMMAPATH</td>
          <td>132</td>
          <td>weight</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>5000</td>
          <td>all</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.LEMMAPATH2.PPMIweight.LENGTH5000.SOCPOSnav</th>
          <td>LEMMAPATH2</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>LEMMAPATH</td>
          <td>132</td>
          <td>weight</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>5000</td>
          <td>nav</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.LEMMAPATH2.PPMIweight.LENGTHFOC.SOCPOSall</th>
          <td>LEMMAPATH2</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>LEMMAPATH</td>
          <td>132</td>
          <td>weight</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>FOC</td>
          <td>all</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.LEMMAPATH2.PPMIweight.LENGTHFOC.SOCPOSnav</th>
          <td>LEMMAPATH2</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>LEMMAPATH</td>
          <td>132</td>
          <td>weight</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>FOC</td>
          <td>nav</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.LEMMAPATH3.PPMIno.LENGTH5000.SOCPOSall</th>
          <td>LEMMAPATH3</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>LEMMAPATH</td>
          <td>250</td>
          <td>no</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>5000</td>
          <td>all</td>
          <td>58</td>
        </tr>
        <tr>
          <th>destroy.LEMMAPATH3.PPMIno.LENGTH5000.SOCPOSnav</th>
          <td>LEMMAPATH3</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>LEMMAPATH</td>
          <td>250</td>
          <td>no</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>5000</td>
          <td>nav</td>
          <td>58</td>
        </tr>
        <tr>
          <th>destroy.LEMMAPATH3.PPMIno.LENGTHFOC.SOCPOSall</th>
          <td>LEMMAPATH3</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>LEMMAPATH</td>
          <td>250</td>
          <td>no</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>FOC</td>
          <td>all</td>
          <td>58</td>
        </tr>
        <tr>
          <th>destroy.LEMMAPATH3.PPMIno.LENGTHFOC.SOCPOSnav</th>
          <td>LEMMAPATH3</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>LEMMAPATH</td>
          <td>250</td>
          <td>no</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>FOC</td>
          <td>nav</td>
          <td>58</td>
        </tr>
        <tr>
          <th>destroy.LEMMAPATH3.PPMIselection.LENGTH5000.SOCPOSall</th>
          <td>LEMMAPATH3</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>LEMMAPATH</td>
          <td>184</td>
          <td>selection</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>5000</td>
          <td>all</td>
          <td>58</td>
        </tr>
        <tr>
          <th>destroy.LEMMAPATH3.PPMIselection.LENGTH5000.SOCPOSnav</th>
          <td>LEMMAPATH3</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>LEMMAPATH</td>
          <td>184</td>
          <td>selection</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>5000</td>
          <td>nav</td>
          <td>58</td>
        </tr>
        <tr>
          <th>destroy.LEMMAPATH3.PPMIselection.LENGTHFOC.SOCPOSall</th>
          <td>LEMMAPATH3</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>LEMMAPATH</td>
          <td>184</td>
          <td>selection</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>FOC</td>
          <td>all</td>
          <td>58</td>
        </tr>
        <tr>
          <th>destroy.LEMMAPATH3.PPMIselection.LENGTHFOC.SOCPOSnav</th>
          <td>LEMMAPATH3</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>LEMMAPATH</td>
          <td>184</td>
          <td>selection</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>FOC</td>
          <td>nav</td>
          <td>58</td>
        </tr>
        <tr>
          <th>destroy.LEMMAPATH3.PPMIweight.LENGTH5000.SOCPOSall</th>
          <td>LEMMAPATH3</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>LEMMAPATH</td>
          <td>184</td>
          <td>weight</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>5000</td>
          <td>all</td>
          <td>58</td>
        </tr>
        <tr>
          <th>destroy.LEMMAPATH3.PPMIweight.LENGTH5000.SOCPOSnav</th>
          <td>LEMMAPATH3</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>LEMMAPATH</td>
          <td>184</td>
          <td>weight</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>5000</td>
          <td>nav</td>
          <td>58</td>
        </tr>
        <tr>
          <th>destroy.LEMMAPATH3.PPMIweight.LENGTHFOC.SOCPOSall</th>
          <td>LEMMAPATH3</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>LEMMAPATH</td>
          <td>184</td>
          <td>weight</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>FOC</td>
          <td>all</td>
          <td>58</td>
        </tr>
        <tr>
          <th>destroy.LEMMAPATH3.PPMIweight.LENGTHFOC.SOCPOSnav</th>
          <td>LEMMAPATH3</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>LEMMAPATH</td>
          <td>184</td>
          <td>weight</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>FOC</td>
          <td>nav</td>
          <td>58</td>
        </tr>
        <tr>
          <th>destroy.LEMMAPATHweight.PPMIno.LENGTH5000.SOCPOSall</th>
          <td>LEMMAPATHweight</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>LEMMAPATH</td>
          <td>250</td>
          <td>no</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>5000</td>
          <td>all</td>
          <td>58</td>
        </tr>
        <tr>
          <th>destroy.LEMMAPATHweight.PPMIno.LENGTH5000.SOCPOSnav</th>
          <td>LEMMAPATHweight</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>LEMMAPATH</td>
          <td>250</td>
          <td>no</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>5000</td>
          <td>nav</td>
          <td>58</td>
        </tr>
        <tr>
          <th>destroy.LEMMAPATHweight.PPMIno.LENGTHFOC.SOCPOSall</th>
          <td>LEMMAPATHweight</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>LEMMAPATH</td>
          <td>250</td>
          <td>no</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>FOC</td>
          <td>all</td>
          <td>58</td>
        </tr>
        <tr>
          <th>destroy.LEMMAPATHweight.PPMIno.LENGTHFOC.SOCPOSnav</th>
          <td>LEMMAPATHweight</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>LEMMAPATH</td>
          <td>250</td>
          <td>no</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>FOC</td>
          <td>nav</td>
          <td>58</td>
        </tr>
        <tr>
          <th>destroy.LEMMAPATHweight.PPMIselection.LENGTH5000.SOCPOSall</th>
          <td>LEMMAPATHweight</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>LEMMAPATH</td>
          <td>184</td>
          <td>selection</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>5000</td>
          <td>all</td>
          <td>58</td>
        </tr>
        <tr>
          <th>destroy.LEMMAPATHweight.PPMIselection.LENGTH5000.SOCPOSnav</th>
          <td>LEMMAPATHweight</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>LEMMAPATH</td>
          <td>184</td>
          <td>selection</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>5000</td>
          <td>nav</td>
          <td>58</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>destroy.nobound3-3lex.PPMIselection.LENGTHFOC.SOCPOSall</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>False</td>
          <td>BOW</td>
          <td>96</td>
          <td>selection</td>
          <td>lex</td>
          <td>3-3</td>
          <td>FOC</td>
          <td>all</td>
          <td>55</td>
        </tr>
        <tr>
          <th>destroy.nobound3-3lex.PPMIselection.LENGTHFOC.SOCPOSnav</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>False</td>
          <td>BOW</td>
          <td>96</td>
          <td>selection</td>
          <td>lex</td>
          <td>3-3</td>
          <td>FOC</td>
          <td>nav</td>
          <td>55</td>
        </tr>
        <tr>
          <th>destroy.nobound3-3lex.PPMIweight.LENGTH5000.SOCPOSall</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>False</td>
          <td>BOW</td>
          <td>96</td>
          <td>weight</td>
          <td>lex</td>
          <td>3-3</td>
          <td>5000</td>
          <td>all</td>
          <td>55</td>
        </tr>
        <tr>
          <th>destroy.nobound3-3lex.PPMIweight.LENGTH5000.SOCPOSnav</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>False</td>
          <td>BOW</td>
          <td>96</td>
          <td>weight</td>
          <td>lex</td>
          <td>3-3</td>
          <td>5000</td>
          <td>nav</td>
          <td>55</td>
        </tr>
        <tr>
          <th>destroy.nobound3-3lex.PPMIweight.LENGTHFOC.SOCPOSall</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>False</td>
          <td>BOW</td>
          <td>96</td>
          <td>weight</td>
          <td>lex</td>
          <td>3-3</td>
          <td>FOC</td>
          <td>all</td>
          <td>55</td>
        </tr>
        <tr>
          <th>destroy.nobound3-3lex.PPMIweight.LENGTHFOC.SOCPOSnav</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>False</td>
          <td>BOW</td>
          <td>96</td>
          <td>weight</td>
          <td>lex</td>
          <td>3-3</td>
          <td>FOC</td>
          <td>nav</td>
          <td>55</td>
        </tr>
        <tr>
          <th>destroy.nobound5-5all.PPMIno.LENGTH5000.SOCPOSall</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>False</td>
          <td>BOW</td>
          <td>283</td>
          <td>no</td>
          <td>all</td>
          <td>5-5</td>
          <td>5000</td>
          <td>all</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.nobound5-5all.PPMIno.LENGTH5000.SOCPOSnav</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>False</td>
          <td>BOW</td>
          <td>283</td>
          <td>no</td>
          <td>all</td>
          <td>5-5</td>
          <td>5000</td>
          <td>nav</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.nobound5-5all.PPMIno.LENGTHFOC.SOCPOSall</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>False</td>
          <td>BOW</td>
          <td>283</td>
          <td>no</td>
          <td>all</td>
          <td>5-5</td>
          <td>FOC</td>
          <td>all</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.nobound5-5all.PPMIno.LENGTHFOC.SOCPOSnav</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>False</td>
          <td>BOW</td>
          <td>283</td>
          <td>no</td>
          <td>all</td>
          <td>5-5</td>
          <td>FOC</td>
          <td>nav</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.nobound5-5all.PPMIselection.LENGTH5000.SOCPOSall</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>False</td>
          <td>BOW</td>
          <td>215</td>
          <td>selection</td>
          <td>all</td>
          <td>5-5</td>
          <td>5000</td>
          <td>all</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.nobound5-5all.PPMIselection.LENGTH5000.SOCPOSnav</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>False</td>
          <td>BOW</td>
          <td>215</td>
          <td>selection</td>
          <td>all</td>
          <td>5-5</td>
          <td>5000</td>
          <td>nav</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.nobound5-5all.PPMIselection.LENGTHFOC.SOCPOSall</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>False</td>
          <td>BOW</td>
          <td>215</td>
          <td>selection</td>
          <td>all</td>
          <td>5-5</td>
          <td>FOC</td>
          <td>all</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.nobound5-5all.PPMIselection.LENGTHFOC.SOCPOSnav</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>False</td>
          <td>BOW</td>
          <td>215</td>
          <td>selection</td>
          <td>all</td>
          <td>5-5</td>
          <td>FOC</td>
          <td>nav</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.nobound5-5all.PPMIweight.LENGTH5000.SOCPOSall</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>False</td>
          <td>BOW</td>
          <td>215</td>
          <td>weight</td>
          <td>all</td>
          <td>5-5</td>
          <td>5000</td>
          <td>all</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.nobound5-5all.PPMIweight.LENGTH5000.SOCPOSnav</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>False</td>
          <td>BOW</td>
          <td>215</td>
          <td>weight</td>
          <td>all</td>
          <td>5-5</td>
          <td>5000</td>
          <td>nav</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.nobound5-5all.PPMIweight.LENGTHFOC.SOCPOSall</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>False</td>
          <td>BOW</td>
          <td>215</td>
          <td>weight</td>
          <td>all</td>
          <td>5-5</td>
          <td>FOC</td>
          <td>all</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.nobound5-5all.PPMIweight.LENGTHFOC.SOCPOSnav</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>False</td>
          <td>BOW</td>
          <td>215</td>
          <td>weight</td>
          <td>all</td>
          <td>5-5</td>
          <td>FOC</td>
          <td>nav</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.nobound5-5lex.PPMIno.LENGTH5000.SOCPOSall</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>False</td>
          <td>BOW</td>
          <td>194</td>
          <td>no</td>
          <td>lex</td>
          <td>5-5</td>
          <td>5000</td>
          <td>all</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.nobound5-5lex.PPMIno.LENGTH5000.SOCPOSnav</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>False</td>
          <td>BOW</td>
          <td>194</td>
          <td>no</td>
          <td>lex</td>
          <td>5-5</td>
          <td>5000</td>
          <td>nav</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.nobound5-5lex.PPMIno.LENGTHFOC.SOCPOSall</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>False</td>
          <td>BOW</td>
          <td>194</td>
          <td>no</td>
          <td>lex</td>
          <td>5-5</td>
          <td>FOC</td>
          <td>all</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.nobound5-5lex.PPMIno.LENGTHFOC.SOCPOSnav</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>False</td>
          <td>BOW</td>
          <td>194</td>
          <td>no</td>
          <td>lex</td>
          <td>5-5</td>
          <td>FOC</td>
          <td>nav</td>
          <td>60</td>
        </tr>
        <tr>
          <th>destroy.nobound5-5lex.PPMIselection.LENGTH5000.SOCPOSall</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>False</td>
          <td>BOW</td>
          <td>153</td>
          <td>selection</td>
          <td>lex</td>
          <td>5-5</td>
          <td>5000</td>
          <td>all</td>
          <td>59</td>
        </tr>
        <tr>
          <th>destroy.nobound5-5lex.PPMIselection.LENGTH5000.SOCPOSnav</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>False</td>
          <td>BOW</td>
          <td>153</td>
          <td>selection</td>
          <td>lex</td>
          <td>5-5</td>
          <td>5000</td>
          <td>nav</td>
          <td>59</td>
        </tr>
        <tr>
          <th>destroy.nobound5-5lex.PPMIselection.LENGTHFOC.SOCPOSall</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>False</td>
          <td>BOW</td>
          <td>153</td>
          <td>selection</td>
          <td>lex</td>
          <td>5-5</td>
          <td>FOC</td>
          <td>all</td>
          <td>59</td>
        </tr>
        <tr>
          <th>destroy.nobound5-5lex.PPMIselection.LENGTHFOC.SOCPOSnav</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>False</td>
          <td>BOW</td>
          <td>153</td>
          <td>selection</td>
          <td>lex</td>
          <td>5-5</td>
          <td>FOC</td>
          <td>nav</td>
          <td>59</td>
        </tr>
        <tr>
          <th>destroy.nobound5-5lex.PPMIweight.LENGTH5000.SOCPOSall</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>False</td>
          <td>BOW</td>
          <td>153</td>
          <td>weight</td>
          <td>lex</td>
          <td>5-5</td>
          <td>5000</td>
          <td>all</td>
          <td>59</td>
        </tr>
        <tr>
          <th>destroy.nobound5-5lex.PPMIweight.LENGTH5000.SOCPOSnav</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>False</td>
          <td>BOW</td>
          <td>153</td>
          <td>weight</td>
          <td>lex</td>
          <td>5-5</td>
          <td>5000</td>
          <td>nav</td>
          <td>59</td>
        </tr>
        <tr>
          <th>destroy.nobound5-5lex.PPMIweight.LENGTHFOC.SOCPOSall</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>False</td>
          <td>BOW</td>
          <td>153</td>
          <td>weight</td>
          <td>lex</td>
          <td>5-5</td>
          <td>FOC</td>
          <td>all</td>
          <td>59</td>
        </tr>
        <tr>
          <th>destroy.nobound5-5lex.PPMIweight.LENGTHFOC.SOCPOSnav</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>False</td>
          <td>BOW</td>
          <td>153</td>
          <td>weight</td>
          <td>lex</td>
          <td>5-5</td>
          <td>FOC</td>
          <td>nav</td>
          <td>59</td>
        </tr>
      </tbody>
    </table>
    <p>204 rows × 11 columns</p>
    </div>



.. code:: ipython3

    socdata.to_csv(f"{github_type}/{type_name}.models.tsv", sep = "\t", index_label="_model")

6 Cosine distances
~~~~~~~~~~~~~~~~~~

Once we have all the token-level vectors, as well as our registers, we
can quickly compute and store their cosine distances.

.. code:: ipython3

    from qlvl import TypeTokenMatrix
    from qlvl.specutils.mxcalc import compute_distance
    
    input_suffix = ".tcmx.soc.pac" #token by context matrix
    output_suffix = ".ttmx.dist.pac" # token by token matrix
    for modelname in socdata.index:
        input_name = f"{token_dir}/{modelname}{input_suffix}"
        output_name = f"{token_dir}/{modelname}{output_suffix}"
        compute_distance(TypeTokenMatrix.load(input_name)).save(output_name)
        


.. parsed-literal::

    
    ************************************
    function    = compute_distance
      time      = 0.004143 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH2.PPMIno.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003516 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH2.PPMIno.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.000495 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH2.PPMIno.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0003891 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH2.PPMIno.LENGTHFOC.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003544 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH2.PPMIselection.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003281 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH2.PPMIselection.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.000458 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH2.PPMIselection.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0003853 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH2.PPMIselection.LENGTHFOC.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003346 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH2.PPMIweight.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.00337 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH2.PPMIweight.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0004458 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH2.PPMIweight.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0003784 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH2.PPMIweight.LENGTHFOC.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003365 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH3.PPMIno.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003258 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH3.PPMIno.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0005183 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH3.PPMIno.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0004103 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH3.PPMIno.LENGTHFOC.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003415 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH3.PPMIselection.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003288 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH3.PPMIselection.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0004792 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH3.PPMIselection.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0003912 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH3.PPMIselection.LENGTHFOC.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003437 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH3.PPMIweight.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003254 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH3.PPMIweight.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0004752 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH3.PPMIweight.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0003858 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATH3.PPMIweight.LENGTHFOC.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003299 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATHweight.PPMIno.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003211 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATHweight.PPMIno.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0005355 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATHweight.PPMIno.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0004344 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATHweight.PPMIno.LENGTHFOC.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003389 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATHweight.PPMIselection.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003386 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATHweight.PPMIselection.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0005322 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATHweight.PPMIselection.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0004265 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATHweight.PPMIselection.LENGTHFOC.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003646 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATHweight.PPMIweight.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003224 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATHweight.PPMIweight.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0007229 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATHweight.PPMIweight.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0004065 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAPATHweight.PPMIweight.LENGTHFOC.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003201 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL1.PPMIno.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.002952 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL1.PPMIno.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0004129 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL1.PPMIno.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.000366 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL1.PPMIno.LENGTHFOC.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.002757 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL1.PPMIselection.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.002507 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL1.PPMIselection.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0003536 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL1.PPMIselection.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0003238 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL1.PPMIselection.LENGTHFOC.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.002454 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL1.PPMIweight.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.002723 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL1.PPMIweight.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0003536 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL1.PPMIweight.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0003459 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL1.PPMIweight.LENGTHFOC.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003461 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL2.PPMIno.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003282 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL2.PPMIno.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0004413 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL2.PPMIno.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0004644 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL2.PPMIno.LENGTHFOC.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003202 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL2.PPMIselection.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003114 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL2.PPMIselection.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.000397 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL2.PPMIselection.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0003636 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL2.PPMIselection.LENGTHFOC.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003042 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL2.PPMIweight.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003112 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL2.PPMIweight.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0004306 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL2.PPMIweight.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0003653 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.LEMMAREL2.PPMIweight.LENGTHFOC.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003399 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10all.PPMIno.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003464 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10all.PPMIno.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0005765 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10all.PPMIno.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0004613 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10all.PPMIno.LENGTHFOC.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003397 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10all.PPMIselection.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003299 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10all.PPMIselection.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0005322 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10all.PPMIselection.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0004106 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10all.PPMIselection.LENGTHFOC.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003435 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10all.PPMIweight.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.00338 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10all.PPMIweight.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0005286 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10all.PPMIweight.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0004089 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10all.PPMIweight.LENGTHFOC.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003426 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10lex.PPMIno.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003166 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10lex.PPMIno.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0005918 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10lex.PPMIno.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0006363 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10lex.PPMIno.LENGTHFOC.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003223 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10lex.PPMIselection.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003161 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10lex.PPMIselection.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0004637 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10lex.PPMIselection.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0004041 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10lex.PPMIselection.LENGTHFOC.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003235 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10lex.PPMIweight.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003181 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10lex.PPMIweight.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0004592 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10lex.PPMIweight.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0004089 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound10-10lex.PPMIweight.LENGTHFOC.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.00348 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3all.PPMIno.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003246 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3all.PPMIno.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.000474 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3all.PPMIno.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0003781 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3all.PPMIno.LENGTHFOC.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003321 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3all.PPMIselection.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003282 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3all.PPMIselection.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0006359 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3all.PPMIselection.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0003681 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3all.PPMIselection.LENGTHFOC.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003399 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3all.PPMIweight.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003199 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3all.PPMIweight.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.000479 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3all.PPMIweight.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0003886 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3all.PPMIweight.LENGTHFOC.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003286 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3lex.PPMIno.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003129 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3lex.PPMIno.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0004051 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3lex.PPMIno.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0003839 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3lex.PPMIno.LENGTHFOC.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003026 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3lex.PPMIselection.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.002897 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3lex.PPMIselection.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0004036 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3lex.PPMIselection.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0003817 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3lex.PPMIselection.LENGTHFOC.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003142 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3lex.PPMIweight.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.002828 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3lex.PPMIweight.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0003819 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3lex.PPMIweight.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0003791 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound3-3lex.PPMIweight.LENGTHFOC.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003356 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5all.PPMIno.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003342 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5all.PPMIno.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.000514 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5all.PPMIno.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0004277 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5all.PPMIno.LENGTHFOC.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003691 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5all.PPMIselection.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003298 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5all.PPMIselection.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0005 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5all.PPMIselection.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0004089 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5all.PPMIselection.LENGTHFOC.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003434 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5all.PPMIweight.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003274 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5all.PPMIweight.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0004692 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5all.PPMIweight.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0004139 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5all.PPMIweight.LENGTHFOC.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003472 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5lex.PPMIno.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003121 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5lex.PPMIno.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0004709 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5lex.PPMIno.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0004172 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5lex.PPMIno.LENGTHFOC.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003232 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5lex.PPMIselection.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003103 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5lex.PPMIselection.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0004814 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5lex.PPMIselection.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0003915 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5lex.PPMIselection.LENGTHFOC.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003195 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5lex.PPMIweight.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003125 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5lex.PPMIweight.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0004191 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5lex.PPMIweight.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0003898 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.bound5-5lex.PPMIweight.LENGTHFOC.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003507 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10all.PPMIno.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003379 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10all.PPMIno.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0007288 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10all.PPMIno.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0005741 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10all.PPMIno.LENGTHFOC.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003446 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10all.PPMIselection.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003355 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10all.PPMIselection.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0005732 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10all.PPMIselection.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0004704 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10all.PPMIselection.LENGTHFOC.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003484 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10all.PPMIweight.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003368 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10all.PPMIweight.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0006027 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10all.PPMIweight.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0004728 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10all.PPMIweight.LENGTHFOC.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.00352 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10lex.PPMIno.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003428 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10lex.PPMIno.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0005722 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10lex.PPMIno.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0005023 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10lex.PPMIno.LENGTHFOC.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003341 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10lex.PPMIselection.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003174 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10lex.PPMIselection.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.000572 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10lex.PPMIselection.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0004327 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10lex.PPMIselection.LENGTHFOC.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003267 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10lex.PPMIweight.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003198 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10lex.PPMIweight.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0005162 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10lex.PPMIweight.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0004287 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound10-10lex.PPMIweight.LENGTHFOC.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003349 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3all.PPMIno.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003366 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3all.PPMIno.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0004704 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3all.PPMIno.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0003893 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3all.PPMIno.LENGTHFOC.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003459 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3all.PPMIselection.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003452 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3all.PPMIselection.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.000452 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3all.PPMIselection.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0004146 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3all.PPMIselection.LENGTHFOC.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0034 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3all.PPMIweight.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003451 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3all.PPMIweight.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0004478 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3all.PPMIweight.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0004036 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3all.PPMIweight.LENGTHFOC.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.00325 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3lex.PPMIno.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003035 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3lex.PPMIno.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.000561 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3lex.PPMIno.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0003879 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3lex.PPMIno.LENGTHFOC.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003215 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3lex.PPMIselection.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.002839 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3lex.PPMIselection.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0004032 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3lex.PPMIselection.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0003643 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3lex.PPMIselection.LENGTHFOC.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.002883 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3lex.PPMIweight.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.002881 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3lex.PPMIweight.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0003841 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3lex.PPMIweight.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0003648 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound3-3lex.PPMIweight.LENGTHFOC.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003575 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5all.PPMIno.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003583 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5all.PPMIno.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0005362 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5all.PPMIno.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0004344 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5all.PPMIno.LENGTHFOC.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003551 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5all.PPMIselection.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003377 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5all.PPMIselection.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0005312 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5all.PPMIselection.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0004342 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5all.PPMIselection.LENGTHFOC.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003565 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5all.PPMIweight.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003304 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5all.PPMIweight.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0004959 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5all.PPMIweight.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0003963 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5all.PPMIweight.LENGTHFOC.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003453 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5lex.PPMIno.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003228 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5lex.PPMIno.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0004728 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5lex.PPMIno.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0004549 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5lex.PPMIno.LENGTHFOC.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003266 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5lex.PPMIselection.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003123 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5lex.PPMIselection.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0004463 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5lex.PPMIselection.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0004165 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5lex.PPMIselection.LENGTHFOC.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003367 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5lex.PPMIweight.LENGTH5000.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.003142 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5lex.PPMIweight.LENGTH5000.SOCPOSnav.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0006673 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5lex.PPMIweight.LENGTHFOC.SOCPOSall.ttmx.dist.pac
    
    ************************************
    function    = compute_distance
      time      = 0.0003977 sec
    ************************************
    
    
    Saving matrix...
    Stored in file:
      /home/projects/semmetrix/mariana_wolken/cleanWorkflow//output//tokens/destroy/destroy.nobound5-5lex.PPMIweight.LENGTHFOC.SOCPOSnav.ttmx.dist.pac


For the rest, we go to R!

The R code is in the processClouds notebook, which uses the
`semcloud <https://github.com/montesmariana/semcloud>`__ package. I plan
to incorporate small clouds into the package to use for examples and
recreate the processClouds notebook as a vignette for the package.

Bonus: context word detail
--------------------------

.. code:: ipython3

    from semasioFlow.contextwords import listContextwords

.. code:: ipython3

    cws = listContextwords(type_name, tokenlist, fnameSample, settings, left_win=15, right_win = 15)


.. parsed-literal::

    100%|██████████| 60/60 [00:00<00:00, 455.16it/s]


.. code:: ipython3

    cw_fname = f"{output_path}/registers/{type_name}.cws.detail.tsv"
    cws.to_csv(cw_fname, sep = "\t", index_label = "cw_id")
    cws

From this table, it is relatively straightforward to extract
concordances and highlight the context words that match certain filters.
Note that by default the left contexts are in reverse order.
