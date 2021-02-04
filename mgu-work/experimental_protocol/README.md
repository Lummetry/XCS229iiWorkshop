conda install nbconvert
conda install -c conda-forge miktex


# BASIC 

  "title" : "Experimental Protocol",
  "authors": [
    {
      "name": "Andrei Damian"
    }
  ],
  
  
# ADVANCED: 

!jupyter nbconvert experiment.ipynb --to latex --template citations.tplx 
!pdflatex --quiet experiment.tex
!bibtex experiment
!pdflatex --quiet experiment.tex
!pdflatex --quiet experiment.tex
!pdflatex --quiet experiment.tex

############## citations.tplx:

((*- extends 'article.tplx' -*))

((* block author *))
\author{Andrei Damian}
((* endblock author *))

((* block title *))
\title{Experimental Protocol}
((* endblock title *))

((* block bibliography *))
\bibliographystyle{unsrt}
\bibliography{jupyter}
((* endblock bibliography *))


########## jupyter.bib



@article{miller1995wordnet,
  title={WordNet: a lexical database for English},
  author={Miller, George A},
  journal={Communications of the ACM},
  volume={38},
  number={11},
  pages={39--41},
  year={1995},
  publisher={ACM New York, NY, USA}
}


@article{mikolov2013efficient,
  title={Efficient estimation of word representations in vector space},
  author={Mikolov, Tomas and Chen, Kai and Corrado, Greg and Dean, Jeffrey},
  journal={arXiv preprint arXiv:1301.3781},
  year={2013}
}