(define (problem tmp)
	(:domain workspace)
	(:objects
		rack - receptacle
		hook - tool
		yogurt - box
		salt - box
	)
	(:init
		(aligned rack)
		(poslimit rack)
		(beyondworkspace rack)
		(on rack table)
		(on hook rack)
		(on salt rack)
		(on yogurt rack)
	)
	(:goal (and))
)
