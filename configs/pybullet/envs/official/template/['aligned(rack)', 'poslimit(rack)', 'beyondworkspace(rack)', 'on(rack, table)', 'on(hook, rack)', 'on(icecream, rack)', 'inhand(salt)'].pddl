(define (problem tmp)
	(:domain workspace)
	(:objects
		rack - receptacle
		hook - tool
		icecream - box
		salt - box
	)
	(:init
		(aligned rack)
		(poslimit rack)
		(beyondworkspace rack)
		(on rack table)
		(on hook rack)
		(on icecream rack)
		(inhand salt)
	)
	(:goal (and))
)
