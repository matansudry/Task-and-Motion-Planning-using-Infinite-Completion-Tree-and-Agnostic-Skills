(define (problem tmp)
	(:domain workspace)
	(:objects
		rack - receptacle
		hook - tool
	)
	(:init
		(aligned rack)
		(poslimit rack)
		(beyondworkspace rack)
		(on rack table)
		(on hook table)
	)
	(:goal (and))
)
