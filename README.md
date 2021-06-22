# PlantGrowthModeling

Firstly, this project is largely inspired and informed by the research paper 
Dynamical Modeling of Plant Growth (N. Bessenov and V. Volpert). The model comes from their papers, as do the numbers used for the parameters. The code and numerical schemes to solve the equations did not come from this paper.

Programs Included:

Plant1: Fully operational; describes plant growth with a constant concentration of metabolites.  
Plant2: Partially operational; describes plant growth with metabolite concentration varying over space and time.   
Plant3: Partially operational; same as Plant2 but with runs significantly faster.  
Plant4: Fully operational; describes plant growth with a predetermined, varying with time concentration of metabolites.  
Diff_Advc_Concept: Fully operational, proof of concept of the diffusion-advection equation, solved and animated.  

User Instructions:
- Open file from Final Project -> Code into Python developer environment.  
- Run file and fill in user inputs. Suggested for inputs are provided at the top of each file. Further comments in the code provide more explanation and allow for more variability in inputs.  

Other Files Included:

Preparation: includes project proposal and research paper (with highlights).  
  
Math: word documents / pdf equivalents of the math done to solve the diffusion-advection equation. The LaxWendroff file is an attempt that turned out to be a dead-end, but shows an interesting effort nonetheless. The Crank_Nicolson file is the solution that was implemented.  
  
Presentation: contains the powerpoint file that summarizes the project, used for in-class presentation.
