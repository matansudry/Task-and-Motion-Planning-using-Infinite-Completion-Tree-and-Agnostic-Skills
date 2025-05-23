(define (problem tmp)
	(:domain workspace)
	(:objects
		rack - receptacle
		hook - tool
		salt - box
	)
	(:init
		(aligned rack)
		(poslimit rack)
		(beyondworkspace rack)
		(on rack table)
		(on hook table)
		(inhand salt)
	)
	(:goal (and))
)
