```
This documents the version control naming syntaxes and it primarily divides the workflow into testphases(T), runningdrafts(D), Current(C)
```

## testphases:
```
The writing format will be vT*, in the primary numerical order (1,2,3....) this holds untested features and designs which are yet to be verified and tested and determined
if the features set would really be pushed out into runningdrafts

    This stage documentation is mainly focused on the feature itself and not on the full scale design yet, this is purely and ideation drawing board
```
--------------------------------------------------------------------------------------------------------

## runningdrafts
```
The writing format will be vD*, in the primary numerical order(1,2,3,....) this holds the features and designs and ideas from testphases which are pushed into
runningdrafts

    1.This stage documentation will hold the features integration with the Current archietecture design, test results, unit test results, the implementation
    design schema itself,

    2. All the revisions and changes will orient on the principal of the narrow vision(1 Draft at a time) to make things manageable, that means before the current Draft
    has not been discarded or pushed successfully into Current no new draft will be allowed to be formed

    3. The documentation must record the change phase and their results on the overall pipeline ie the draft will be acting as implementation drawing board
    and thus there will always be a copy of the current implementation of frame AR to test the integration on, keeping the main build (C) isolated from any collapses
    this means documentation must record the reason for the loss delta in draft if the feature was pushed to current so the design is not unreasonable for a faster push

```
----------------------------------------------------------------------------------------------------------------------------------------------------------------

## Current
```
Holds the stable build release which is what will be accessible whenever running the framework

    This will hold the current implementation documentation and use case implementation
```
