# Survey of Image Style Transfer Methodologies
Group members: mcbaron (solo)

The proposed project is an examination of different methods for image style transfer, from a linear algebraic construction through automatic and convolutional neural networks. I'd like to take a broad survey to examine the similarities and differences between approaches to this question, in order to understand what patterns and features a machine can learn that underlie human image perception.

I propose creating a few constructions of solutions to the problem of transferring style between two images, both for artistic and photo-realistic outcomes. The importance of a survey across methodologies is to help refine the path forward to an algorithmic understanding of how humans create and perceive artistic imagery. A practical application, of course, is (semi-)automatic stylization for casual photographers. A professional photographer will have a compelling visual style that has been refined over years of practice, which an amatuer will simply not have. Additionally, because professional artists sometimes produce entire collections in a common style, the semi-autonomous stylization could be a time saver for professionals looking to reproduce a curated visual style on new images. 

The methods I’d like to pursue are outlined in a few papers:
[Style Transfer Via Image Component Analysis](http://ieeexplore.ieee.org/document/6522845/) by Wei Zhang, Chen Cao, Shifeng Chen, Jianzhuang Liu
[Style Transfer for Headshot Portraits](https://people.csail.mit.edu/yichangshih/portrait_web/2014_portrait.pdf) by YiChang Shih, Sylvain Paris, Connelly Barnes, William T. Freeman, and Frédo Durand
[A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576) by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge
[Deep Photo Style Transfer](https://arxiv.org/abs/1703.07511) by Fujun Luan, Sylvain Paris, Eli Shechtman, Kavita Bala

I seek to implement all of these methods for output comparison and methodological understanding, but factoring in possible time constraints, I believe a realistic outcome is comparison of 2-3 methods. I would also like to attempt a naive approach using a Total Variation minimization approach, which would be an original solution to this problem. I do not expect my approach to be nearly as fruitful as previous solutions, but it will serve as a point of comparison. 

I will be reliant on the datasets that are cited by the aforementioned works. Many of which have already been segmented for photo-realistic style transfer. Furthermore, utilizing known datasets will allow for direct comparison of results from my implementations and that of the original authors. For a demonstration during a poster presentation, I hope to use user submitted photos to demonstrate the methods live. 
