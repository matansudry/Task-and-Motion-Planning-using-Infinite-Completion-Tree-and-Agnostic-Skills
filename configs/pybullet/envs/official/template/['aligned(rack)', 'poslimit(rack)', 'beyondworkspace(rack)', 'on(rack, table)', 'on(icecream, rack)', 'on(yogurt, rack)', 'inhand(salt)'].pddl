(define (problem tmp)
	(:domain workspace)
	(:objects
		rack - receptacle
		yogurt - box
		icecream - box
		salt - box
	)
	(:init
		(aligned rack)
		(poslimit rack)
		(beyondworkspace rack)
		(on rack table)
		(on icecream rack)
		(on yogurt rack)
		(inhand salt)
	)
	(:goal (and))
)
