# Scalable Distributed String Sorting

This is source code for my master's thesis on _Scalable Distributed String Sorting Algorithms_ at
the Karlsruhe Institute of Technology --- Institute of Theoretical Informatics, Algorithm Engineering.

> String sorting algorithms have been studied extensively for sequential and shared-memory parallel
>     models of computation.
> There has however been comparatively little and only very recent work covering string sorting in
>     distributed-memory parallel systems.
> In this thesis, we directly build on the existing work to develop distributed algorithms that are
>     more scalable with respect to two parameters: the number of processors used for sorting and the
>     input size per processor in terms of characters.
> For the first aspect, we develop a multi-level generalization of existing multi-way string merge
>     sort, based on a technique that has been used successfully in atomic sorting.
> The developed algorithm is experimentally demonstrated to perform well for a range of inputs across
>     a spectrum of magnitudes.
> We observe speedups up to five over the closest existing competitor on up to \num{24576}
>     processors.
> 
> To make distributed string sorting more scalable with respect to input size, we develop a
>     space-efficient sorting framework which primarily distinguishes itself through the use of a
>     compressed input format.
> By deduplicating overlapping substrings and sorting the input in smaller chunks rather than as a
>     whole, it is possible to create sorted permutations for inputs that would otherwise exceed the
>     available memory.
> We experimentally confirm this claim by demonstrating that an implementation of the framework is
>     able to sort inputs containing \qty{22.4}{\giga\byte} uncompressed characters per processor with
>     only \qty{2}{\giga\byte} memory available on average.
> Furthermore, an application of space-efficient sorting in suffix array construction, specifically
>     as subroutine to DCX, is proposed.
> We show that our implementation is capable of sorting large difference covers for texts containing
>     up to \qty{1.23}{\tera\byte} in characters.
