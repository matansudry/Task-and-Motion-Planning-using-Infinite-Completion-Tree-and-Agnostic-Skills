(define (problem tmp)
	(:domain workspace)
	(:objects
		rack - receptacle
		hook - tool
		milk - box
		icecream - box
	)
	(:init
		(aligned rack)
		(poslimit rack)
		(inworkspace rack)
		(on rack table)
		(on hook table)
		(on milk table)
		(on icecream table)
	)
	(:goal (and))
)
