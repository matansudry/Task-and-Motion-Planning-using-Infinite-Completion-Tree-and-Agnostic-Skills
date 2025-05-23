(define (problem tmp)
	(:domain workspace)
	(:objects
		rack - receptacle
		hook - tool
		icecream - box
	)
	(:init
		(aligned rack)
		(poslimit rack)
		(inworkspace rack)
		(on rack table)
		(on hook table)
		(on icecream rack)
	)
	(:goal (and))
)
