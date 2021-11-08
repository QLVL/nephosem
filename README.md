
![Release](https://img.shields.io/github/v/release/qlvl/nephosem)
[![License](https://img.shields.io/github/license/qlvl/nephosem)](https://www.gnu.org/licenses/gpl-3.0)


This is a Python module to create count-based distributional models for semantic analysis. It was developed within the [Nephological Semantics project](https://www.arts.kuleuven.be/ling/qlvl/projects/current/nephological-semantics) at KU Leuven, mostly written by [Tao Chen](https://github.com/enzocxt/) and with the collaboration of Dirk Geeraerts, Dirk Speelman, Kris Heylen, Weiwei Zhang, Karlien Franco, Stefano De Pascale and [Mariana Montes](https://github.com/montesmariana/).

The code can be implemented but still requires thorough automatic testing tools.

# Installation and use

In order to use this code, clone this repository, add it to your PATH and then import the `nephosem` library:

```python
import os
os.path.append('/path/to/repository')
import nephosem
```

<!-- Here we can add a link to the documentation, tutorials, my repositories with my own python/R code... -->
<!-- For a semasiological perspective like the one followed [here](https://cloudspotting.marianamontes.me/), you can follow...  -->
<!-- For an onomasiological/lectometric perspective... -->

# Background

Schütze, Hinrich. 1998. Automatic Word Sense Discrimination. _Computational Linguistics_ 24(1). 97–123.
<!-- Any other suggestions? -->

# Publications using this code

De Pascale, S. 2019. _Token-based vector space models as semantic control in lexical lectometry_. Leuven: KU Leuven PhD Dissertation. (8 November, 2019).

De Pascale, Stefano & Weiwei Zhang. 2021. Scoring with Token-based Models. A Distributional Semantic Replication of Socioectometric Analyses in Geeraerts, Grondelaers, and Speelman (1999). In Gitte Kristiansen, Karlien Franco, Stefano De Pascale, Laura Rosseel & Weiwei Zhang (eds.), _Cognitive Sociolinguistics Revisited_, 186–199. De Gruyter. https://doi.org/10.1515/9783110733945-021.

Montes, Mariana. 2021. _Cloudspotting: visual analytics for distributional semantics_. Leuven: KU Leuven PhD Dissertation.

Montes, Mariana, Karlien Franco & Kris Heylen. 2021. Indestructible Insights. A Case Study in Distributional Prototype Semantics. In Gitte Kristiansen, Karlien Franco, Stefano De Pascale, Laura Rosseel & Weiwei Zhang (eds.), _Cognitive Sociolinguistics Revisited_, 251–263. De Gruyter. https://doi.org/10.1515/9783110733945-021.

Montes, Mariana & Kris Heylen. 2022. Visualizing Distributional Semantics. In Dennis Tay & Molly Xie Pan (eds.), _Data Analytics in Cognitive Linguistics. Methods and Insights_. Mouton De Gruyter.
<!-- We should soon be able to add the publications in Cognitive Sociolinguistics Revisited -->

# Related publications

Heylen, Kris, Dirk Speelman & Dirk Geeraerts. 2012. Looking at word meaning. An interactive visualization of Semantic Vector Spaces for Dutch synsets. In _Proceedings of the eacl 2012 Joint Workshop of LINGVIS & UNCLH_, 16–24. Avignon.

Heylen, Kris, Thomas Wielfaert, Dirk Speelman & Dirk Geeraerts. 2015. Monitoring polysemy: Word space models as a tool for large-scale lexical semantic analysis. _Lingua_ 157. 153–172.

Speelman, Dirk, Stefan Grondelaers, Benedikt Szmrecsanyi & Kris Heylen. 2020. Schaalvergroting in het syntactische alternantieonderzoek: Een nieuwe analyse van het presentatieve er met automatisch gegenereerde predictoren. _Nederlandse Taalkunde_ 25(1). 101–123. https://doi.org/10.5117/NEDTAA2020.1.005.SPEE.
